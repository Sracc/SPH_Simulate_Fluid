import taichi as ti
from functools import reduce
import numpy as np
import trimesh as tm


@ti.data_oriented
class ParticleSystem:
    def __init__(self, res):
        # Control rigid motion
        self.start = ti.field(ti.i32, shape=())
        self.end = ti.field(ti.i32, shape=())
        self.up = ti.field(ti.i32, shape=())
        self.up[None] = 0
        self.res = res
        # particle parameter
        self.particle_num = ti.field(ti.i32, shape=())
        self.positions = ti.Vector.field(3, dtype=ti.f32)
        self.particle_velocity = ti.Vector.field(3, dtype=ti.f32)
        self.particle_density = ti.field(dtype=float)
        self.gravity = ti.Vector([0.0, 0.0, -9.81])
        self.up_gravity = ti.Vector([0.0, 0.0, 9.81])
        # particle radius dx
        self.particle_radius = 0.1
        self.particle_diameter = 2 * self.particle_radius
        # kernel radius dh
        self.dh = self.particle_radius * 1.4
        self.nonepressure_acc = ti.Vector.field(3, dtype=ti.f32)
        self.particle_positions = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.predicted_position = ti.Vector.field(3, dtype=ti.f32)
        self.predicted_velocity = ti.Vector.field(3, dtype=ti.f32)
        self.particle_pressure = ti.field(dtype=float)
        self.particle_pressure_acc = ti.Vector.field(3, dtype=ti.f32)
        self.d_density = ti.field(ti.f32)
        # Grid variables
        self.grid_size = self.dh * 2
        self.grid_num = np.ceil(np.array(res) / self.grid_size).astype(int)
        self.fluid_particle = 1
        self.rigid_particle = 2
        self.material = ti.field(dtype=int)
        # system bound
        self.bound = [res[1], 0, 0, res[0], res[2], 0]
        self.top_bound = self.bound[4]  # top_bound
        self.bottom_bound = self.bound[5]  # bottom_bound
        self.left_bound = self.bound[2]  # left_bound
        self.right_bound = self.bound[3]  # right_bound
        self.front_bound = self.bound[0]
        self.back_bound = self.bound[1]

        particles_node = ti.root.dense(ti.i, 2 ** 18)  # dense data structure
        particles_node.place(self.positions, self.predicted_position, self.particle_pressure, self.predicted_velocity,
                             self.particle_density, self.particle_pressure_acc, \
                             self.nonepressure_acc, self.particle_velocity, self.d_density, self.material, )

    @ti.func
    def validate_particle_type(self, particle):
        # check particle type
        return self.material[particle]

    # Detect particle interaction boundaries
    @ti.kernel
    def detect_boundary(self):
        for p_i in range(self.particle_num[None]):
            if self.validate_particle_type(p_i) == self.fluid_particle:
                particle_position = self.positions[p_i]
                if particle_position[0] < self.left_bound + 0.5 * self.particle_diameter:
                    self.simulate_collisions(
                        p_i, ti.Vector([1, 0.0, 0.0]),
                        self.left_bound + 0.5 * self.particle_diameter - particle_position[0])
                if particle_position[0] > self.right_bound - 0.5 * self.particle_diameter:
                    self.simulate_collisions(
                        p_i, ti.Vector([-1, 0.0, 0.0]),
                        particle_position[0] - self.right_bound + 0.5 * self.particle_diameter)
                if particle_position[2] > self.top_bound - 0.5 * self.particle_diameter:
                    self.simulate_collisions(
                        p_i, ti.Vector([0.0, 0.0, -1]),
                        particle_position[2] - self.top_bound + 0.5 * self.particle_diameter)
                if particle_position[2] < self.bottom_bound + 0.5 * self.particle_diameter:
                    self.simulate_collisions(
                        p_i, ti.Vector([0.0, 0.0, 1]),
                        self.bottom_bound + 0.5 * self.particle_diameter - particle_position[2])
                if particle_position[1] > self.front_bound - 0.5 * self.particle_diameter:
                    self.simulate_collisions(
                        p_i, ti.Vector([0.0, -1, 0.0]),
                        particle_position[1] - self.front_bound + 0.5 * self.particle_diameter)
                if particle_position[1] < self.back_bound + 0.5 * self.particle_diameter:
                    self.simulate_collisions(
                        p_i, ti.Vector([0.0, 1, 0.0]),
                        self.back_bound + 0.5 * self.particle_diameter - particle_position[1])
            elif self.validate_particle_type(p_i) == self.rigid_particle:
                particle_position = self.positions[p_i]
                if particle_position[2] > self.top_bound - 0.5 * self.particle_diameter:
                    for i in range(self.start[None], self.start[None] + self.end[None]):
                        self.particle_velocity[i] *= ti.Vector([0, 0.0, -0.0])
                    self.up[None] = 0
                elif particle_position[2] < self.bottom_bound + 0.5 * self.particle_diameter:
                    for i in range(self.start[None], self.start[None] + self.end[None]):
                        self.particle_velocity[i] *= ti.Vector([0.0, 0.0, -0.0])
                    self.up[None] = 1

    @ti.func
    def simulate_collisions(self, p_i, vec, d):
        self.positions[p_i] += vec * d
        self.particle_velocity[p_i] -= (1 + 0.5) * self.particle_velocity[p_i].dot(vec) * vec

    # generate a cube with neatly arranged particles
    def add_cube(self,
                 lower_corner,
                 cube_size,
                 material,
                 ):
        num_dim = []
        for i in range(3):
            num_dim.append(np.arange(lower_corner[i], lower_corner[i] + cube_size[i] + 0.5, self.particle_radius))
        num_new_particles = reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])
        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(
            -1, reduce(lambda x, y: x * y, list(new_positions.shape[1:])))
        print(new_positions)
        self.add_particle_cube(num_new_particles, new_positions, material)
        # Add to current particles count
        self.particle_num[None] += num_new_particles


    @ti.func
    def initialize_particle(self, index, position, res):
        self.positions[index] = position
        self.predicted_position[index] = position
        self.particle_velocity[index] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        self.predicted_velocity[index] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        self.nonepressure_acc[index] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        self.particle_pressure[index] = 0.0
        self.particle_pressure_acc[index] = ti.Vector(
            [0.0, 0.0, 0.0], dt=ti.f32)
        self.particle_density[index] = 1000.0
        self.d_density[index] = 0.0
        self.material[index] = self.fluid_particle
        n = 0
        if res == 1:
            self.material[index] = self.rigid_particle
            self.particle_density[index] = 1000.0
            n += 1

    @ti.kernel
    def add_particle_cube(self, new_particles: ti.i32, new_positions: ti.types.ndarray(),
                          new_material: ti.i32):
        if new_material == self.rigid_particle:
            self.start[None] = self.particle_num[None]
            self.end[None] = new_particles
        for i in range(self.particle_num[None], self.particle_num[None] + new_particles):
            self.material[i] = new_material
            position = ti.Vector([0.0, 0.0, 0.0])
            for k in ti.static(range(3)):
                position[k] = new_positions[k, i - self.particle_num[None]]
            res = 0
            if new_material == self.fluid_particle:
                res = 0
            else:
                res = 1
            self.initialize_particle(i, position, res)

        print(self.start[None], self.end[None], new_particles)

    @ti.kernel
    def copy_dynamic_nd(self, np_x: ti.types.ndarray(), input_x: ti.template()):
        for i in range(self.particle_num[None]):
            for j in ti.static(range(3)):
                np_x[i, j] = input_x[i][j]

    @ti.kernel
    def copy_dynamic(self, np_x: ti.types.ndarray(), input_x: ti.template()):
        for i in range(self.particle_num[None]):
            np_x[i] = input_x[i]

    def particle_info(self):
        particle_position = np.ndarray((self.particle_num[None], 3),
                                       dtype=np.float32)
        self.copy_dynamic_nd(particle_position, self.positions)
        particle_velocity = np.ndarray((self.particle_num[None], 3),
                                       dtype=np.float32)
        self.copy_dynamic_nd(particle_velocity, self.particle_velocity)
        particle_material = np.ndarray((self.particle_num[None],), dtype=np.int32)
        self.copy_dynamic(particle_material, self.material)
        return {
            'position': particle_position,
            'velocity': particle_velocity,
            'material': particle_material
        }

    # Convert the obj or ply model to a particle model
    def add_model(self, model_path, model_scale, offset, pitch, material):
        mesh = tm.load(model_path)
        mesh_scale = model_scale
        mesh.apply_scale(mesh_scale)
        offset = np.array(offset, dtype=np.float32)
        tm.repair.fill_holes(mesh)
        voxelized_mesh = mesh.voxelized(pitch=pitch).fill()
        voxelized_points_np = voxelized_mesh.points + offset
        num_particles_obj = voxelized_points_np.shape[0]
        voxelized_points = ti.Vector.field(3, ti.f32, num_particles_obj)
        voxelized_points.from_numpy(np.array(voxelized_points_np, dtype=np.float32))
        self.add_particles(
            num_particles_obj,
            np.array(voxelized_points_np, dtype=np.float32),  # position
            material, )

    def add_particles(self,
                      new_particles_num: ti.i32,
                      new_particles_positions: ti.types.ndarray(ti.f32),
                      new_particles_material: ti.i32,
                      ):
        self.add_particles_taichi(
            new_particles_num,
            new_particles_positions,
            new_particles_material,
        )

    @ti.kernel
    def add_particles_taichi(self,
                             new_particles_num: ti.i32,
                             new_particles_positions: ti.types.ndarray(ti.f32),
                             new_particles_material: ti.i32,
                             ):
        # print(new_particles_material)
        res = 1
        if new_particles_material == self.rigid_particle:
            res = 1
            self.start[None] = self.particle_num[None]
            self.end[None] = new_particles_num
        elif new_particles_material == self.fluid_particle:
            res = 0
        for particle_index in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            position = ti.Vector([0.0, 0.0, 0.0])
            for d in ti.static(range(3)):
                position[d] = new_particles_positions[particle_index - self.particle_num[None], d]
            self.initialize_particle(particle_index, position, res)
        self.particle_num[None] += new_particles_num
