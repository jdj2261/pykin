import numpy as np
import trimesh
import os

class RobotVisualizer:
    def __init__(self, dh_params):
        """
        Initialize visualizer with DH parameters and STL files path
        dh_params: List of dictionaries with keys ['alpha', 'a', 'd', 'theta']
        visual_path: Path to directory containing STL files
        Note: Length units (a, d) are in millimeters
        """
        self.dh_params = dh_params
        
    def transform_from_dh(self, alpha, a, d, theta):
        """
        Calculate transformation matrix from DH parameters
        Note: a and d should be in millimeters
        """
        # Convert mm to meters for visualization
        a = a / 1000.0
        d = d / 1000.0
        
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct, -st, 0, a],
            [st * ca, ct * ca, -sa, -sa * d],
            [st * sa, ct * sa, ca, ca * d],
            [0, 0, 0, 1]
        ])
    
    def create_coordinate_frame(self, size=0.05):
        """Create coordinate frame mesh"""
        return trimesh.creation.axis(origin_size=size)
        
    def visualize(self, joint_angles):
        """
        Display robot visualization with STL meshes
        Args:
            joint_angles: List of joint angles in radians
        """
        scene = trimesh.Scene()
        
        # Add coordinate frame to show world origin
        scene.add_geometry(self.create_coordinate_frame())
        
        # Initial transformation
        T = np.eye(4)

        # Set numpy print options
        np.set_printoptions(precision=3, suppress=True)

        # Add each link with proper transformation
        for i, (dh, angle) in enumerate(zip(self.dh_params, joint_angles)):
            # Calculate new transformation
            dh_transform = self.transform_from_dh(
                dh['alpha'], dh['a'], dh['d'], angle + dh['theta']
            )
            T = T @ dh_transform
            frame = self.create_coordinate_frame()
            frame.apply_transform(T)
            scene.add_geometry(frame)

        # Show the scene
        scene.show()


# Example usage
if __name__ == "__main__":
    # DH parameters for R-2000iC/165F (all lengths in mm)
    dh_params = [
        {'alpha': 0, 'a': 0, 'd': 670, 'theta': 0},           # 670mm
        {'alpha': np.pi/2, 'a': 312.0, 'd': 0, 'theta': 0},   # 312mm
        {'alpha': 0, 'a': 1075.0, 'd': 0, 'theta': 0},        # 1075mm
        {'alpha': np.pi/2, 'a': 225.0, 'd': 1280.0, 'theta': 0},  # 225mm, 1280mm
        {'alpha': -np.pi/2, 'a': 0, 'd': 0, 'theta': 0},
        {'alpha': np.pi/2, 'a': 0, 'd': 215.0, 'theta': np.pi}    # 215mm
    ]
    
    # Create visualizer
    visualizer = RobotVisualizer(dh_params)
    
    # Test with different joint angles (in radians)
    joint_angles = [0, np.pi/2, 0, 0, 0, 0]  # Home position
    visualizer.visualize(joint_angles)