import taichi as ti
import numpy as np
import math


@ti.data_oriented
class SPHKernel:
    def __init__(self, particle_system):
        self.ps = particle_system

    @ti.func
    def cubic_kernel(self, r, r_norm, h):
        result = ti.cast(0.0, ti.f32)
        # value of cubic spline smoothing kernel
        k = 8 / np.pi / ti.pow(h, 3)
        q = r_norm / h
        if r_norm > 1.0e-5 and q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                result = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                result = k * 2 * ti.pow(1 - q, 3.0)
        return result


    @ti.func
    def derivative_cubic_kernel(self, result_dim, r, r_norm, h):
        result = ti.Vector([0.0, 0.0, 0.0])
        k = 6. * 8 / np.pi / ti.pow(h, 3)
        q = r_norm / h
        if r_norm > 1.0e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)

            if q <= 0.5:
                result = k * q * (3.0 * q - 2.0) * grad_q
            elif q <= 1:
                result = -1.0 * k * ti.pow(1.0 - q, 2) * grad_q
        return result

    @ti.func
    def density_derivative_cubic_kernel(self, r, r_norm, h):
        result = ti.cast(0.0, ti.f32)
        k = 6. * 8 / np.pi / ti.pow(h, 3)
        q = r_norm / h
        if r_norm > 1.0e-5 and q <= 1.0:
            if q <= 0.5:
                result = k * q * (3.0 * q - 2.0)
            else:
                result = -1.0 * k * ti.pow(1.0 - q, 2)
        return result

    @ti.func
    def spiky_kernel(self, r, h):
        result = ti.Vector([0.0, 0.0, 0.0])
        if r >= h:
            result = 0.0
        else:
            x1 = 1.0 - r / h
            result = -45.0 / (np.pi * ti.pow(h, 4)) * x1 * x1
        return result

    @ti.func
    def derivative_spiky_kernel(self, r_norm, h):
        result = 0.0
        q = r_norm / h
        if 0 < r_norm and r_norm < h:
            x = (h * h - r_norm * r_norm) / (h * h * h)
            result = -45.0 / np.pi * (h ** 6) * (h - r_norm) ** 2
        else:
            result = 0.0
        return result

    @ti.func
    def cohesion_kernel(self, r):

        m_k = 32.0 / (np.pi * ti.pow(self.ps.dh, 9.0))
        m_c = ti.pow(self.ps.dh, 6.0) / 64.0
        res = 0.0
        radius2 = self.ps.dh * self.ps.dh
        r2 = r * r

        if r2 <= radius2:
            r3 = r2 * r
            if r > 0.5 * self.ps.dh:
                res = m_k * ti.pow(self.ps.dh - r, 3.0) * r3
            else:
                res = m_k * 2.0 * ti.pow(self.ps.dh - r, 3.0) * r3 - m_c
        return res

    @ti.func
    def poly6_kernel(self, r_norm, h):
        result = 0.0
        if 0 < r_norm and r_norm < h:
            x = (h * h - r_norm * r_norm) / (h * h * h)
            result = 315.0 / 64.0 / np.pi * x * x * x
        return result

    @ti.func
    def derivative_poly6_kernel(self, r_norm, h):
        result = 0.0
        if 0 < r_norm and r_norm < h:
            q = r_norm / h
            x = 3 * (h ** 2 - r_norm ** 2) ** 2 - 4 * r_norm * r_norm * (h ** 2 - r_norm ** 2)
            result = -945.0 / 32.0 / np.pi * (h ** 9) * x * q
        else:
            result = 0.0
        return result

    @ti.func
    def derivative_viscosity_kernel(self, r_norm, h):

        result = 0.0
        if 0 < r_norm and r_norm < h:
            q = r_norm / h

            result = 45 * (h - r_norm) / np.pi * (h ** 6)
        else:
            result = 0.0
        return result

