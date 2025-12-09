"""
Camera module for capturing and processing images from the simulation.
"""

import numpy as np
from typing import Tuple, Dict, Any


class Camera:
    """
    Camera interface for the simulation.
    
    Captures RGB and depth images from the PyBullet camera.
    """
    
    def __init__(self, environment):
        """
        Initialize camera with simulation environment.
        
        Args:
            environment: SimulationEnvironment instance
        """
        self.env = environment
        self.width = environment.camera_width
        self.height = environment.camera_height
        self.intrinsics = environment.get_camera_intrinsics()
        
    def capture(self) -> Dict[str, np.ndarray]:
        """
        Capture a frame from the camera.
        
        Returns:
            Dictionary containing:
            - 'rgb': (H, W, 3) uint8 array
            - 'depth': (H, W) float32 array
            - 'segmentation': (H, W) int32 array
        """
        rgb, depth, seg = self.env.capture_image()
        
        return {
            'rgb': rgb,
            'depth': depth,
            'segmentation': seg,
        }
    
    def get_intrinsics(self) -> np.ndarray:
        """
        Get the camera intrinsic matrix.
        
        Returns:
            3x3 intrinsic matrix K
        """
        return self.intrinsics
    
    def get_view_matrix(self) -> np.ndarray:
        """
        Get the camera view matrix (extrinsics).
        
        Returns:
            4x4 view matrix
        """
        return np.array(self.env.view_matrix).reshape(4, 4).T
    
    def get_projection_matrix(self) -> np.ndarray:
        """
        Get the camera projection matrix.
        
        Returns:
            4x4 projection matrix
        """
        return np.array(self.env.projection_matrix).reshape(4, 4).T
