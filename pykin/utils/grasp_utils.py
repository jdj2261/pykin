import numpy as np
import trimesh
np.seterr(divide='ignore', invalid='ignore')

class GraspManager:
    def __init__(self, gripper=None):
        if gripper is not None:
            self.gripper = gripper
            self.grasp_c_manager = trimesh.collision.CollisionManager()

    def compute_grasp_pose(self):
        pass

    def compute_robust_force_closure(self, mesh, vertices, normals, limit_radian=0.02, n_trials=5):
        sigma = 1e-3
        noise = np.random.normal(0, sigma, (n_trials, 2, 3))
        
        count = 0
        for i in range(n_trials):
            # vertices_copy = vertices.copy()
            new_vertices = vertices + noise[i]

            points, _, faces = trimesh.proximity.closest_point(mesh, new_vertices)        
            normals = mesh.face_normals[faces]

            is_fc = self.is_force_closure(points, normals, limit_radian)
            if is_fc:
                count += 1
        return count/n_trials

    def is_force_closure(self, vertices, normals, limit_radian=0.05):
        normal0 = -1.0 * normals[0] / (1.0 * np.linalg.norm(normals[0]))
        normal1 = -1.0 * normals[1] / (1.0 * np.linalg.norm(normals[1]))

        alpha = np.arctan(limit_radian)
        line = vertices[0] - vertices[1]
        
        line = line / np.linalg.norm(line)
        angle1 = np.arccos(normal1.dot(line))

        line = -1 * line
        angle2 = np.arccos(normal0.dot(line))


        if angle1 > alpha or angle2 > alpha:
            return False
        return True

    @staticmethod
    def surface_sampling(mesh, n_samples=2):
        vertices, face_ind = trimesh.sample.sample_surface(mesh, count=n_samples)
        normals = mesh.face_normals[face_ind]
        return vertices, normals

    @staticmethod
    def find_intersections(mesh, p1, p2):
        pass
    
    @staticmethod
    def visualize_grasp_point():
        pass

# if __name__ == "__main__":
#     fig, ax = plt.init_3d_figure(figsize=(10,6), dpi= 100)

#     gm = GraspManager()
#     mesh = trimesh.load('../../asset/objects/meshes/box_goal.stl')
#     mesh.apply_scale(0.001)

#     while True:

#         # vertices, normals = gm.surface_sampling(mesh, n_samples=2)
#         # result = gm.compute_robust_force_closure(mesh, vertices, normals)
#         # if result > 0.2:
#         #     break
        
#         vertices, normals = gm.surface_sampling(mesh, n_samples=2)
#         if gm.is_force_closure(vertices, normals, limit_radian=0.02):
#             break

#     offset_pos = np.array([0.5, 0.5, 0.5])
#     vertices = vertices + np.tile(offset_pos, (2,1))
    
#     plt.plot_mesh(ax=ax, mesh=mesh, A2B=Transform(pos=offset_pos).h_mat, alpha=0.1, color=[0.5, 0, 0])
#     plt.plot_vertices(ax, vertices)
#     plt.plot_normal_vector(ax, vertices, -normals, scale=0.1)    
#     plt.show_figure()
    



