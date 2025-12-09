"""
Path generation for surface treatment.

Generates toolpaths for:
- Spiral pattern (for circular defects like rust)
- Raster pattern (for linear defects like cracks)

Includes velocity constraints per Codex feedback.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

from src.config import config
from src.vision.localization import Pose3D


@dataclass
class Waypoint:
    """
    A waypoint in a toolpath.
    
    Attributes:
        position: [x, y, z] in world frame
        orientation: [qx, qy, qz, qw] quaternion
        velocity: Target velocity at this point
    """
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    velocity: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "position": list(self.position),
            "orientation": list(self.orientation),
            "velocity": self.velocity,
        }


class PathGenerator:
    """
    Generates toolpaths for defect repair.
    """
    
    def __init__(self):
        """Initialize with config parameters."""
        path_config = config.get("path", {})
        self.max_velocity = path_config.get("max_velocity", 0.1)
        self.max_acceleration = path_config.get("max_acceleration", 0.05)
    
    def generate_spiral(
        self,
        center: Pose3D,
        radius: float = 0.05,
        spacing: float = 0.01,
        num_loops: int = 2,
        hover_height: float = 0.02
    ) -> List[Waypoint]:
        """
        Generate an Archimedean spiral path centered on a defect.
        
        Args:
            center: Center pose with position, orientation, normal
            radius: Maximum spiral radius
            spacing: Distance between spiral loops
            num_loops: Number of complete loops
            hover_height: Height above surface
            
        Returns:
            List of Waypoints forming the spiral path
        """
        waypoints = []
        
        # Extract center info
        cx, cy, cz = center.position
        nx, ny, nz = center.normal
        
        # Create local coordinate system on the surface
        # Z = surface normal (pointing out)
        z_axis = np.array([nx, ny, nz])
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # X = arbitrary perpendicular vector
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, np.array([0, 0, 1]))
        else:
            x_axis = np.cross(z_axis, np.array([1, 0, 0]))
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y = complete the right-handed system
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Generate spiral points
        total_angle = num_loops * 2 * np.pi
        num_points = int(total_angle / (spacing / radius * 10)) + 1
        num_points = max(num_points, 20)
        
        angles = np.linspace(0, total_angle, num_points)
        
        for i, theta in enumerate(angles):
            # Spiral radius increases with angle
            r = spacing * theta / (2 * np.pi)
            r = min(r, radius)
            
            # Position in local frame
            local_x = r * np.cos(theta)
            local_y = r * np.sin(theta)
            local_z = hover_height  # Above surface
            
            # Transform to world frame
            world_pos = (
                cx + local_x * x_axis[0] + local_y * y_axis[0] + local_z * z_axis[0],
                cy + local_x * x_axis[1] + local_y * y_axis[1] + local_z * z_axis[1],
                cz + local_x * x_axis[2] + local_y * y_axis[2] + local_z * z_axis[2],
            )
            
            waypoints.append(Waypoint(
                position=world_pos,
                orientation=center.orientation,
                velocity=0.0,  # Will be set by time_parameterize
            ))
        
        # Apply velocity profile
        waypoints = self.time_parameterize(waypoints)
        
        return waypoints
    
    def generate_raster(
        self,
        center: Pose3D,
        width: float = 0.08,
        height: float = 0.04,
        spacing: float = 0.01,
        hover_height: float = 0.02
    ) -> List[Waypoint]:
        """
        Generate a raster (back-and-forth) path over a defect.
        
        Args:
            center: Center pose with position, orientation, normal
            width: Width of the raster pattern
            height: Height of the raster pattern
            spacing: Distance between raster lines
            hover_height: Height above surface
            
        Returns:
            List of Waypoints forming the raster path
        """
        waypoints = []
        
        # Extract center info
        cx, cy, cz = center.position
        nx, ny, nz = center.normal
        
        # Create local coordinate system
        z_axis = np.array([nx, ny, nz])
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, np.array([0, 0, 1]))
        else:
            x_axis = np.cross(z_axis, np.array([1, 0, 0]))
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Generate raster lines
        num_lines = int(height / spacing) + 1
        y_positions = np.linspace(-height/2, height/2, num_lines)
        
        for i, y_local in enumerate(y_positions):
            # Alternate direction for each line
            if i % 2 == 0:
                x_range = np.linspace(-width/2, width/2, 10)
            else:
                x_range = np.linspace(width/2, -width/2, 10)
            
            for x_local in x_range:
                local_z = hover_height
                
                world_pos = (
                    cx + x_local * x_axis[0] + y_local * y_axis[0] + local_z * z_axis[0],
                    cy + x_local * x_axis[1] + y_local * y_axis[1] + local_z * z_axis[1],
                    cz + x_local * x_axis[2] + y_local * y_axis[2] + local_z * z_axis[2],
                )
                
                waypoints.append(Waypoint(
                    position=world_pos,
                    orientation=center.orientation,
                    velocity=0.0,
                ))
        
        # Apply velocity profile
        waypoints = self.time_parameterize(waypoints)
        
        return waypoints
    
    def time_parameterize(
        self,
        waypoints: List[Waypoint]
    ) -> List[Waypoint]:
        """
        Add velocity profile with trapezoidal acceleration (per Codex feedback).
        
        Args:
            waypoints: List of waypoints without velocity
            
        Returns:
            Waypoints with velocity assigned
        """
        if len(waypoints) < 2:
            return waypoints
        
        n = len(waypoints)
        
        # Trapezoidal profile: ramp up, cruise, ramp down
        ramp_length = n // 4  # 25% ramp up, 50% cruise, 25% ramp down
        
        for i in range(n):
            if i < ramp_length:
                # Ramp up
                t = i / ramp_length
                vel = t * self.max_velocity
            elif i >= n - ramp_length:
                # Ramp down
                t = (n - 1 - i) / ramp_length
                vel = t * self.max_velocity
            else:
                # Cruise
                vel = self.max_velocity
            
            waypoints[i].velocity = vel
        
        return waypoints
    
    def get_approach_point(
        self,
        target: Pose3D,
        approach_distance: float = 0.1
    ) -> Waypoint:
        """
        Generate an approach point above the target.
        
        Args:
            target: Target pose
            approach_distance: Distance above target for approach
            
        Returns:
            Waypoint for approach position
        """
        nx, ny, nz = target.normal
        normal = np.array([nx, ny, nz])
        normal = normal / np.linalg.norm(normal)
        
        pos = np.array(target.position) + normal * approach_distance
        
        return Waypoint(
            position=tuple(pos),
            orientation=target.orientation,
            velocity=self.max_velocity / 2,  # Slower approach
        )
