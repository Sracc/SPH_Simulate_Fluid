import taichi as ti
import numpy as np
import bpy

import pickle
import os
# FROM TAICHI-BLEND ADDON
import numblend as nb

from SPH_Simulate_Fluid.Particle_System.Particle_System import ParticleSystem
from SPH_Simulate_Fluid.SPH_Math.SPH_Kernel import SPHKernel
from SPH_Simulate_Fluid.Neighbor_Search.Uniform_Grid import UniformGrid
from SPH_Simulate_Fluid.Neighbor_Search.Grid_Hash import HashGrid
from SPH_Simulate_Fluid.SPH_Physics.Compute_Physics_Variable import ComputeVariable
from SPH_Simulate_Fluid.SPH_Physics.WCSPH_Physics import WCSPHPhysics


# -------------------
#  UTILITY FUNCTIONS
# -------------------

def delete_object(name):
    try:
        obj = bpy.data.objects[name]
    except KeyError:
        return False
    bpy.data.objects.remove(obj)


def delete_all_objects():
    # FOR INITIALISING
    for scene_object in bpy.data.objects:
        delete_object(scene_object.name)
    for scene_mesh in bpy.data.meshes:
        delete_object(scene_mesh.name)


# PICKLE
use_cache = False
save_cache = True
save_cache_filename = "1. WCSPH.pkl"

cache_path = r"/Applications/Blender.app/Contents/Resources/2.83/python/lib/python3.7/site-packages/SPH_Simulate_Fluid/WCSPH_Cache"

save_ply_file = False
ply_nick_name = "SPH_CUBE_SEPARATED"

ply_export_path = r"/Applications/Blender.app/Contents/Resources/2.83/python/lib/python3.7/site-packages/SPH_Simulate_Fluid/WCSPH_Ply"


total_simulate_frames = 300


def dump_cache():
    pickle.dump(cache, open(os.path.join(cache_path, save_cache_filename), "wb"))


def load_cache(local_cache_filename):
    return pickle.load(open(os.path.join(cache_path, local_cache_filename), "rb"))
###save ply
def save_ply(frame_number, total_point_count, point_pos_np, ply_name):
    series_prefix = ply_name
    np_pos = point_pos_np
    writer = ti.PLYWriter(num_vertices=total_point_count)
    writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
    writer.export_frame_ascii(frame_number, series_prefix)


def save_ply_separate_material(frame_number, entire_particle_info_for_frame):
    # COUNT THE NUMBER OF PARTICLES FOR EACH MATERIAL
    material_counter = {}

    for i in entire_particle_info_for_frame['material']:
        if i in material_counter:
            material_counter[i] += 1
        else:
            material_counter[i] = 1

    # KEEP TRACK OF MATERIALS
    material_position_dictionary = {}  # STORE LISTS OF PARTICLE POSITIONS

    # CREATE HOLDERS
    for material_id in material_counter.keys():
        particle_count_of_material = material_counter[material_id]
        material_position_dictionary[material_id] = np.zeros((particle_count_of_material, 3), dtype="float32")

    progress_counter = dict.fromkeys(material_counter.keys(), 0)

    # SEPARATE THE PARTICLES BY MATERIAL
    for particle_index in range(len(entire_particle_info_for_frame['material'])):
        current_particle_material = entire_particle_info_for_frame['material'][particle_index]
        current_particle_position = entire_particle_info_for_frame['position'][particle_index]

        vacant_index = progress_counter[current_particle_material]
        material_position_dictionary[current_particle_material][vacant_index] = current_particle_position

        progress_counter[current_particle_material] += 1

    for material_name in material_position_dictionary.keys():
        all_particle_positions_for_material = material_position_dictionary[material_name]

        save_ply(frame_number, material_counter[material_name], all_particle_positions_for_material,
                 os.path.join(ply_export_path, f"{ply_nick_name}_m{material_name}.ply"))

nb.init()
delete_all_objects()

# CREATE NEW MESH AND OBJECT, BASED ON POINT LOCATION
mesh = nb.new_mesh('point_cloud')
object = nb.new_object('point_cloud', mesh)

if use_cache is True:
    cache = load_cache(save_cache_filename)
else:
    # USE TAICHI
    ti.init(arch=ti.gpu)

    res = (5, 5, 16)
    ps = ParticleSystem(res)
    ps.add_cube(lower_corner=[1,2, 1.2],
                    cube_size=[2, 2, 2.0],
                    material=ps.fluid_particle,
                    )

    path = r"/Applications/Blender.app/Contents/Resources/2.83/python/lib/python3.7/site-packages/SPH_Simulate_Fluid/model/shovel.ply"
    model_scale = 10
    offset = [3, 3.5 + 0.2, 0.5]
    pitch = 0.065
    material = ps.rigid_particle
    ps.add_model(path, model_scale, offset, pitch, material)
    sph_kernel = SPHKernel(ps)
    # neighbor_search = HashGrid(ps)
    neighbor_search = UniformGrid(ps)
    compute_variable = ComputeVariable(sph_kernel, ps)
    wcsph_phy = WCSPHPhysics(sph_kernel, ps, neighbor_search, compute_variable)

    mesh = nb.new_mesh('test')
    object = nb.new_object('test', mesh)

    cache = []
    n = 0
    res = 0
    for i in range(total_simulate_frames):
        if n == 200:
            res = 1

        for j in range(60):
            wcsph_phy.step(res)

        # particles = ps.particle_info()
        to_append = ps.particle_info()
        cache.append(to_append)
        n += 1

@nb.add_animation
def main():
    for frame in range(total_simulate_frames):
        # for j in range(10):
        point_position_of_point_cloud = cache[frame]['position']
        yield nb.mesh_update(mesh, point_position_of_point_cloud)


if save_cache is True:
    dump_cache()
if save_ply_file is True:
    for frame in range(len(cache)):
        save_ply_separate_material(frame, cache[frame])