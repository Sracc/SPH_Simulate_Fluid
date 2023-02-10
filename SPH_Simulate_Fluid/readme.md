# Module
This project contains two SPH methods and two neighbor search methods.
Grid_Hash.py, Uniform_Grid.py
WCSPH.Physics.py, PCISPH_Physics.py
The difference between the two neighbor search methods is whether the grid index of the grid table is computed using Hash.
The method of using hash value as grid index reduces the memory consumption, but also increases the computing time.


# Demo in Blender
PCISPH_Blender_StoneSand.py
WCSPH_Blender_ShovelWater.py
**In order to visualize the computing results, the code needs to be run in Blender by Taichi-Blender Addon.**

# Demo "headless test"
PCISPH_Test.py
WCSPH_Test.py
This test is also known as "headless", Just running in Python without visualizing the results

# Cache
The Cache contains particle information about the two demos after computing.

# Error
There are some bugs in Blender Taichi Addon. If the project runs in Blender, there is a small probability of error. 
If there is an error in running, restart Blender.
