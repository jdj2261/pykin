import numpy as np
import trimesh
np.seterr(divide='ignore', invalid='ignore')

class GraspManager:
    def __init__(self, gripper=None, max_width=None):
        if gripper is not None:
            self.gripper = gripper
            self.grasp_c_manager = trimesh.collision.CollisionManager()
        self.max_width = max_width
    
    def compute_grasp_pose(self):
        T = np.eye(4)

        return T

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

    def is_force_closure(self, vertices, normals, limit_angle=0.1):
        vectorA = vertices[0]
        vectorB = vertices[1]

        normalA = -normals[0]
        normalB = -normals[1]

        vectorAB = vectorB - vectorA
        distance = np.linalg.norm(vectorAB)

        unit_vectorAB = GraspManager.normalize(vectorAB)
        angle_A2AB = np.arccos(normalA.dot(unit_vectorAB))

        unit_vectorBA = -1 * unit_vectorAB
        angle_B2AB = np.arccos(normalB.dot(unit_vectorBA))

        if distance > self.max_width:
            return False

        if angle_A2AB > limit_angle or angle_B2AB > limit_angle:
            return False

        return True

    @staticmethod
    def surface_sampling(mesh, n_samples=2):
        vertices, face_ind = trimesh.sample.sample_surface(mesh, count=n_samples)
        normals = mesh.face_normals[face_ind]
        return vertices, normals

    @staticmethod
    def find_intersections(mesh, p1, p2):
        ray_origin = (p1 + p2) / 2
        ray_length = np.linalg.norm(p1 - p2)
        ray_dir = (p2 - p1) / ray_length
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=[ray_origin, ray_origin],
            ray_directions=[ray_dir, -ray_dir],
            multiple_hits=True)
        if len(locations) == 0:
            return [], []
        dist_to_center = np.linalg.norm(locations - ray_origin, axis=1)
        dist_mask = dist_to_center <= (ray_length / 2) # only keep intersections on the segment.
        on_segment = locations[dist_mask]
        faces = index_tri[dist_mask]
        return on_segment, faces

    @staticmethod
    def projection(v, u):
        return np.dot(v, u) / np.dot(u, u) * u

    @staticmethod
    def normalize(vec):
        return vec / np.linalg.norm(vec)

    @staticmethod
    def visualize_grasp_point():
        pass