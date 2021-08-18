
import sys
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt

# trimesh1 = trimesh.load( '../../asset/urdf/baxter/meshes/base/pedestal_link_collision.stl')


# mesh1 = pyrender.Mesh.from_trimesh(trimesh1)

# scene = pyrender.Scene()
# scene.add(mesh1)

# pyrender.Viewer(scene, use_raymond_lighting=True)

# plt.figure()
# plt.subplot(1,2,1)
# plt.axis('off')
# plt.imshow(color)
# plt.subplot(1,2,2)
# plt.axis('off')
# plt.imshow(depth, cmap=plt.cm.gray_r)
# plt.show()

# print(depth)


if __name__ == '__main__':
    # print logged messages
    trimesh.util.attach_to_log()

    # load a mesh
    mesh = trimesh.load(
        '../../asset/urdf/baxter/meshes/base/pedestal_link_collision.stl')

    scene = mesh.scene()
    scene.add_geometry(mesh, transform=np.eye(4))
    scene.show()

