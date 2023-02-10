
import taichi as ti
import sys
from Particle_System.Particle_System import ParticleSystem
from SPH_Math.SPH_Kernel import SPHKernel
from Neighbor_Search.Uniform_Grid import UniformGrid
from Neighbor_Search.Grid_Hash import HashGrid
from SPH_Physics.Compute_Physics_Variable import ComputeVariable
from SPH_Physics.WCSPH_Physics import WCSPHPhysics

from memory_profiler import profile
# clean object
ti.init(arch=ti.gpu)

# test memory consumption from different neighbor search method
@profile
def main():
    #250 frames
    #res = (64, 64, 64)
    res = (32, 32, 32)
    ps = ParticleSystem(res)
    ps.add_cube(lower_corner=[1, 2, 1.2],
                cube_size=[2, 2, 2],
                material=ps.fluid_particle,
                )
    path=r"/Applications/Blender.app/Contents/Resources/2.83/python/lib/python3.7/site-packages/SPH_Simulate_Fluid/model/shovel.ply"
    model_scale=10
    offset=[3, 3.5 + 0.2, 0.5]
    pitch=0.065
    material=ps.rigid_particle
    # ps.add_model(path,model_scale,offset,pitch,material)
    sph_kernel = SPHKernel(ps)
    neighbor_search = UniformGrid(ps)
    # neighbor_search = HashGrid(ps)


    compute_variable = ComputeVariable(sph_kernel, ps)
    wcsph_phy = WCSPHPhysics(sph_kernel, ps, neighbor_search, compute_variable)

    for i in range(50):
        for i in range(60):
            wcsph_phy.step(res=0)
            # particles = ps.particle_info()

if __name__ == '__main__':
    main()