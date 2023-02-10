import taichi as ti
import numpy as np


@ti.data_oriented
class UniformGrid:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.grid_particles_num = ti.field(dtype=ti.i32)
        self.grid2particles = ti.field(dtype=ti.i32)
        self.particle_neighbors = ti.field(dtype=ti.i32)
        self.particle_neighbors_num = ti.field(dtype=ti.i32)
        self.grid_num = np.ceil(np.array(self.ps.res) / self.ps.grid_size).astype(int)
        # grid table
        grid_snode = ti.root.pointer(ti.ijk, self.grid_num)
        grid_snode.place(self.grid_particles_num)
        # Maximum Neighbor 100
        grid_snode.dense(ti.l, 100).place(self.grid2particles)
        # neighbor table
        nb_node = ti.root.dynamic(ti.i, 2 ** 18)
        nb_node.place(self.particle_neighbors_num)
        nb_node.dense(ti.j, 100).place(self.particle_neighbors) 

    @ti.func
    def get_cell_index(self, position):

        # p1 = 73856093 * pos.x
        # p2 = 19349663 * pos.y
        # p3 = 83492791 * pos.z
        # s=int((p1 ^ p2 ^ p3) % (self.ps.particle_num[None] * 2))
        # return [(p1 ^ p2 ^ p3) % (self.ps.particle_num[None] * 2),0,0]
        return (position / (self.ps.dh*2)).cast(int)

    @ti.kernel
    def update_hash_table(self):
        # allocate particles to grid
        for p_i in range(self.ps.particle_num[None]):
            # Compute the grid index
            grid_cell = self.get_cell_index(self.ps.positions[p_i])
            offs = ti.atomic_add(self.grid_particles_num[grid_cell], 1)
            # Maximum Neighbor 100
            if offs < 100:
                self.grid2particles[grid_cell, offs] = p_i

    @ti.func
    def validate_grid(self, range_cell):
        res = 1
        for i in ti.static(range(3)):
            res = ti.atomic_and(res, (0 <= range_cell[i] < self.grid_num[i]))
        return res

    # search neighbors from adjacent cells
    @ti.kernel
    def search_neighbors(self):
        for p_i in range(self.ps.particle_num[None]):
            pos_i = self.ps.positions[p_i]
            nb_i = 0
            cell = self.get_cell_index(self.ps.positions[p_i])
            for offs in ti.static(
                    ti.grouped(ti.ndrange(*((-1, 2),) * 3))):
                range_cell = cell + offs
                if self.validate_grid(range_cell) == 1:
                    for j in range(self.grid_particles_num[range_cell]):
                        p_j = self.grid2particles[range_cell, j]
                        if nb_i < 100 and p_j != p_i and (
                                pos_i - self.ps.positions[p_j]
                        ).norm() < self.ps.dh*2.0:
                            self.particle_neighbors[p_i, nb_i] = p_j
                            nb_i += 1
            self.particle_neighbors_num[p_i] = nb_i

    def initialize_hash_grid(self):
        # clear grid
        self.grid_particles_num.fill(0)
        self.particle_neighbors.fill(-1)
