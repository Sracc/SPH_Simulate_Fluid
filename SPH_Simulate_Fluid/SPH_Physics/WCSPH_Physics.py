import numpy as np
import taichi as ti


@ti.data_oriented
class WCSPHPhysics:
    def __init__(self, sph_math, particle_system, grid_search, compute_variable):
        self.sph_math = sph_math
        self.ps = particle_system
        self.grid_s = grid_search
        self.c_variable = compute_variable
        self.rho_0 = 1000.0  # rest density
        self.viscosity = 0.9
        self.surface_tension = 0.01
        # time step
        self.dt = ti.field(ti.f32, shape=())
        self.dt.from_numpy(np.array(0.00015, dtype=np.float32))

    @ti.kernel
    def compute_nonpressure_acceleration_density(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.validate_particle_type(p_i) == self.ps.fluid_particle:
                pos_i = self.ps.positions[p_i]
                d_v = ti.Vector([0.0, 0.0, 0.0])
                d_rho = 0.0
                # search neighbor compute ∑j
                for j in range(self.grid_s.particle_neighbors_num[p_i]):
                    p_j = self.grid_s.particle_neighbors[p_i, j]
                    pos_j = self.ps.positions[p_j]
                    if self.ps.validate_particle_type(p_j) == self.ps.fluid_particle:
                        # Compute position ij
                        r = pos_i - pos_j
                        if r.norm() > 1.0e-5:
                            vij = self.ps.particle_velocity[p_i]-self.ps.particle_velocity[p_j]
                            # compute derivative density
                            d_rho += self.c_variable.compute_derivative_density(r, vij)
                            # self.ps.particle_density[p_i]=d_rho*self.dt[None]
                            diameter2 = self.ps.particle_diameter ** 2
                            r2 = r.dot(r)
                            if r2 > diameter2:
                                # Dvi =−αmi ∑mj􏰊xi−xj􏰋Wij,
                                d_v += self.c_variable.compute_surface_tension(r, r.norm(), self.surface_tension)
                            else:
                                d_v += self.c_variable.\
                                    compute_surface_tension(r, ti.Vector([self.ps.particle_diameter, 0.0, 0.0]).norm(), self.surface_tension)
                            d_v += self.c_variable.viscosity_force(p_j, vij, r, self.viscosity)
                    elif self.ps.validate_particle_type(p_j) == self.ps.rigid_particle:
                        r = pos_i - pos_j
                        if r.norm() > 1.0e-5:
                            vij = self.ps.particle_velocity[p_i] - self.ps.particle_velocity[p_j]
                            d_rho += self.c_variable.compute_derivative_density(r, vij)
                d_v += self.ps.gravity
                self.ps.nonepressure_acc[p_i] = d_v
                self.ps.d_density[p_i] = d_rho

    @ti.kernel
    def compute_total_pressure_acceleration(self, res: ti.int32):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.validate_particle_type(p_i) == self.ps.fluid_particle:
                self.ps.particle_density[p_i] += self.dt[None] * self.ps.d_density[p_i]
                self.ps.particle_density[p_i] = ti.max(self.ps.particle_density[p_i], self.rho_0)
                self.ps.particle_pressure[p_i] = self.compute_pressure_tait(
                    self.ps.particle_density[p_i], self.rho_0, 7.0,
                    20)

                pos_i = self.ps.positions[p_i]
                for j in range(self.grid_s.particle_neighbors_num[p_i]):
                    p_j = self.grid_s.particle_neighbors[p_i, j]
                    pos_j = self.ps.positions[p_j]
                    r = pos_i-pos_j
                    if self.ps.validate_particle_type(p_j) == self.ps.fluid_particle:
                        self.ps.nonepressure_acc[p_i] += \
                            self.c_variable.compute_pressure_acceleration(p_i, p_j, r, r.norm(), 0)
                    elif self.ps.validate_particle_type(p_j) == self.ps.rigid_particle:
                        self.ps.nonepressure_acc[p_i] += \
                            self.c_variable.compute_pressure_acceleration(p_i, p_j, r, r.norm(), 0)
            elif self.ps.validate_particle_type(p_i) == self.ps.rigid_particle:
                # Control the solid to move up and down according to the number of frames
                if res == 1:
                    if self.ps.up[None] == 0:
                        self.ps.nonepressure_acc[p_i] = self.ps.gravity
                    else:
                        self.ps.nonepressure_acc[p_i] = self.ps.up_gravity

    @ti.kernel
    def update_next_time_step(self):
        for p_i in range(self.ps.particle_num[None]):
            self.c_variable.compute_velocity_position(p_i, self.dt[None])
            
    @ti.func
    def compute_pressure_tait(self, rho, rho_0=1000.0, gamma=7.0, c_0=20):
        # Weakly compressible, tait function
        b = rho_0 * (c_0 ** 2) / gamma
        return b * ((rho / rho_0) ** gamma - 1.0)

    def update_neighbor(self):
        self.grid_s.initialize_hash_grid()
        self.grid_s.update_hash_table()
        self.grid_s.search_neighbors()

    def step(self, res):
        # neighbor search
        self.update_neighbor()
        # WCSPH core
        self.compute_nonpressure_acceleration_density()
        self.compute_total_pressure_acceleration(res)
        self.update_next_time_step()
        # simulate collision
        self.ps.detect_boundary()


