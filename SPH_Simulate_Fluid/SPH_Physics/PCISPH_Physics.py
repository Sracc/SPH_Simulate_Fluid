import taichi as ti
import numpy as np


@ti.data_oriented
class PCISPHPhysics:
    def __init__(self, sph_math, particle_system, grid_search, compute_variable):
        self.sph_math = sph_math
        self.ps = particle_system
        self.grid_s = grid_search
        self.c_variable = compute_variable
        self.num_iteration = 0
        self.least_iteration = 5
        self.max_rho_err = ti.field(dtype=ti.f32, shape=())
        self.rho_0 = 1000.0
        # time step
        self.dt = ti.field(ti.f32, shape=())
        self.dt.from_numpy(np.array(0.00075, dtype=np.float32))

        self.viscosity = 1.5
        self.surface_tension = 0.5
        self.density_error = ti.field(ti.f32, shape=())
        self.kpci = ti.field(ti.f32, shape=())
        self.kpci.from_numpy(np.array(1, dtype=np.float32))

    @ti.kernel
    def compute_nonpressure_force(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.validate_particle_type(p_i) == self.ps.fluid_particle:
                pos_i = self.ps.positions[p_i]  # position
                vel_i = self.ps.particle_velocity[p_i]  # velocity
                d_v = ti.Vector([0.0, 0.0, 0.0])
                for j in range(self.grid_s.particle_neighbors_num[p_i]):
                    p_j = self.grid_s.particle_neighbors[p_i, j]
                    if self.ps.validate_particle_type(p_j) == self.ps.fluid_particle:
                        pos_j = self.ps.positions[p_j]
                        vel_j = self.ps.particle_velocity[p_j]

                        x_ij = pos_i - pos_j
                        v_ij = vel_i - vel_j
                        diameter2 = self.ps.particle_diameter * self.ps.particle_diameter

                        r2 = x_ij.dot(x_ij)
                        if r2 > diameter2:
                            d_v += self.c_variable.compute_surface_tension(x_ij, x_ij.norm(), self.surface_tension)
                        else:
                            d_v += self.c_variable.compute_surface_tension(x_ij, ti.Vector([self.ps.particle_diameter, 0.0, 0.0]).norm(), self.surface_tension)
                        if x_ij.norm() > 0:
                            d_v += self.c_variable.viscosity_force(p_j, v_ij, x_ij, self.viscosity)
                d_v += self.ps.gravity
                self.ps.nonepressure_acc[p_i] = d_v
                self.ps.particle_pressure[p_i] = 0.0
                self.ps.particle_pressure_acc[p_i] = ti.Vector(
                    [0.0, 0.0, 0.0])

    @ti.kernel
    def predict_velocity_position(self):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.validate_particle_type(p_i) == self.ps.fluid_particle:
                self.ps.predicted_velocity[p_i] = self.ps.particle_velocity[p_i] + \
                                                  (self.ps.nonepressure_acc[p_i] + self.ps.particle_pressure_acc[p_i]) * \
                                                  self.dt[None]
                self.ps.predicted_position[p_i] = self.ps.positions[p_i] + self.ps.predicted_velocity[p_i] * self.dt[
                    None]
        self.max_rho_err[None] = 0.0
        # Initialize the max_rho_err

    # kPCi = -1 / β(−Σj∇Wij⋅Σj∇Wij−Σj(∇Wij⋅∇Wij)) β=0.5(rho_0)**2/dt*m**2
    @ti.kernel
    def compute_kpci_factor(self):
        grad_sum = ti.Vector([0.0, 0.0, 0.0])
        grad_dot_sum = 0.0
        # perfect sampling
        range_num = ti.cast(self.ps.dh * 2.0 / self.ps.particle_radius, ti.i32)
        half_range = ti.cast(range_num * 0.5, ti.i32)
        for i_x in range(-half_range, half_range):
            for i_y in range(-half_range, half_range):
                for i_z in range(-half_range, half_range):
                    r = ti.Vector([-i_x * self.ps.particle_radius, -i_y * self.ps.particle_radius,
                                   -i_z * self.ps.particle_radius])
                    if self.ps.dh * 2.0 >= r.norm() >= 1e-5:
                        grad = self.sph_math.derivative_cubic_kernel(3, r, r.norm(), self.ps.dh)
                        grad_sum += grad
                        grad_dot_sum += grad.dot(grad)
        beta = 2 * (self.dt[None] * self.c_variable.m / self.rho_0) ** 2
        self.kpci[None] = 1.0 / (beta * (grad_sum.dot(grad_sum) + grad_dot_sum))

    @ti.kernel
    def update_pressure_density_error(self):
        # predict density, density error, update pressure
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.validate_particle_type(p_i) == self.ps.fluid_particle:
                pos_i = self.ps.predicted_position[p_i]  # position
                vel_i = self.ps.predicted_velocity[p_i]
                d_rho = 0.0
                for j in range(self.grid_s.particle_neighbors_num[p_i]):
                    p_j = self.grid_s.particle_neighbors[p_i, j]
                    pos_j = self.ps.predicted_position[p_j]
                    vel_j = self.ps.predicted_velocity[p_j]
                    x_ij = pos_i - pos_j
                    v_ij = vel_i - vel_j
                    if x_ij.norm() > 1e-5:
                        # compute derivative density
                        d_rho += self.c_variable.compute_derivative_density(x_ij, v_ij)
                self.ps.d_density[p_i] = d_rho * self.dt[None]
                self.density_error[None] = self.ps.particle_density[p_i] + self.ps.d_density[p_i] - self.rho_0
                self.ps.particle_pressure[p_i] += self.kpci[None] * (self.density_error[None])
                self.max_rho_err[None] = max(abs(self.density_error[None]), self.max_rho_err[None])

    @ti.kernel
    def update_pressure_acceleration(self, res: ti.int32):
        self.max_rho_err[None] = 0.0
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.validate_particle_type(p_i) == self.ps.fluid_particle:
                pos_i = self.ps.predicted_position[p_i]
                pressure_acc = ti.Vector([0.0, 0.0, 0.0])
                for j in range(self.grid_s.particle_neighbors_num[p_i]):
                    p_j = self.grid_s.particle_neighbors[p_i, j]
                    pos_j = self.ps.predicted_position[p_j]
                    r = pos_i - pos_j
                    # Avoid particle coincidence
                    if r.norm() > 1e-5:
                        # Compute Pressure force acceleration
                        pressure_acc += self.c_variable.compute_pressure_acceleration(p_i, p_j, r, r.norm(), 1)
                self.ps.particle_pressure_acc[p_i] = pressure_acc
            elif self.ps.validate_particle_type(p_i) == self.ps.rigid_particle:
                # Control the solid to move up and down according to the number of frames
                if res == 1:
                    if self.ps.up[None] == 0:
                        self.ps.nonepressure_acc[p_i] = self.ps.gravity
                    else:
                        self.ps.nonepressure_acc[p_i] = self.ps.up_gravity

    @ti.kernel
    def update_velocity_position(self):
        # velocity, position and density
        for p_i in range(self.ps.particle_num[None]):
            self.c_variable.compute_velocity_position(p_i, self.dt[None])
            self.ps.particle_density[p_i] += self.ps.d_density[p_i]
            self.ps.particle_density[p_i] = ti.max(self.ps.particle_density[p_i], self.rho_0)

    def update_neighbor(self):
        self.grid_s.initialize_hash_grid()
        self.grid_s.update_hash_table()
        self.grid_s.search_neighbors()

    def compute_kpci_constant(self):
        self.compute_kpci_factor()

    def step(self, res):
        self.update_neighbor()
        self.compute_nonpressure_force()
        self.num_iteration = 0  # number of iteration
        # Iterative solution of pressure to redudce density error
        while self.max_rho_err[None] > 0.01 * self.rho_0 or self.num_iteration < self.least_iteration:
            self.predict_velocity_position()
            self.update_pressure_density_error()
            self.update_pressure_acceleration(res)
            self.num_iteration += 1
            if self.num_iteration > 1000:
                print("Error: The number of iteration exceeds 1000 ")
                break
        self.update_velocity_position()
        self.ps.detect_boundary()
