import taichi as ti


@ti.data_oriented
class ComputeVariable:
    def __init__(self, sph_math, particle_system):
        self.sph_math = sph_math
        self.ps = particle_system
        self.rho_0 = 1000.0  # rest density
        self.m_V = self.ps.particle_radius ** 3  # volume
        self.m = self.m_V * self.rho_0  # mass

    # compute viscosity force Equation ∇2vi =2(d+2)∑mj (vij·xij)/(∥xij∥2 +0.01h2)∇Wij
    @ti.func
    def viscosity_force(self, j, v_ij, r, viscosity):
        r_mod = r.norm()
        # 0.9is particle viscous force parameters
        result = 2 * (3 + 2) * viscosity * (self.m / (self.ps.particle_density[j])) * v_ij.dot(r) / (
                r_mod ** 2 + 0.01 * self.ps.dh ** 2) * \
                 self.sph_math.derivative_cubic_kernel(3, r, r_mod, self.ps.dh)

        return result

    # ap=−∑mj(pi/ρi**2+pj/ρj**2)∇Wij ≈−mi(2pi/p0**2)∑j∇Wij
    @ti.func
    def compute_pressure_acceleration(self, p_i, p_j, r, r_norm, approximate):
        result = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)
        if approximate == 0:
            result = -self.m * (self.ps.particle_pressure[p_i] / self.ps.particle_density[p_i] ** 2
                                + self.ps.particle_pressure[p_j] / self.ps.particle_density[p_j] ** 2) \
                     * self.sph_math.derivative_cubic_kernel(3, r, r_norm, self.ps.dh)
        else:
            # ap=≈−mi(2ρi/ρ0**2)∑j∇Wij
            result = -self.m * (self.ps.particle_pressure[p_i] * 2.0 / self.rho_0 ** 2) \
                     * self.sph_math.derivative_cubic_kernel(3, r, r_norm, self.ps.dh)
        return result

    # Dρi/Dt=∑jmj(vi-vj)*∇Wij
    @ti.func
    def compute_derivative_density(self, r, vij):
        return self.m * self.sph_math.density_derivative_cubic_kernel(r, r.norm(), self.ps.dh) \
               * vij.dot(r / r.norm())

    #  Dvi/Dt =−α/mi∑jmj(xi−xj)Wij
    @ti.func
    def compute_surface_tension(self, r, r_norm, surface_tension):
        return (-surface_tension / self.m) * self.m * r * self.sph_math.cubic_kernel(
            r, r_norm, self.ps.dh)

    @ti.func
    def compute_velocity_position(self, p_i, dt):
        self.ps.particle_velocity[p_i] += dt * (self.ps.nonepressure_acc[p_i] + self.ps.particle_pressure_acc[p_i])
        self.ps.positions[
            p_i] += dt * self.ps.particle_velocity[p_i]
