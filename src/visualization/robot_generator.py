"""
Sci-Fi Scout Drone Robot Generator

Generates a procedural 3D robot drone using composite geometric meshes:
- Main chassis (ellipsoid)
- Armor plating (side panels)
- Articulated legs with joints
- Antennae and thrusters

Usage:
    from src.visualization.robot_generator import get_robot_figure, SciFiDrone
    
    fig = get_robot_figure()
    fig.show()
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Optional
from dataclasses import dataclass


# ============ COLOR PALETTE ============
COLORS = {
    'chassis_white': 'rgb(224, 224, 224)',      # #E0E0E0
    'gunmetal': 'rgb(112, 112, 112)',           # #707070
    'industrial_yellow': 'rgb(255, 193, 7)',    # #FFC107
    'dark_grey': 'rgb(64, 64, 64)',             # Joint color
    'sensor_dark': 'rgb(32, 32, 32)',           # Eye/sensor
    'electric_blue': 'rgb(0, 255, 255)',        # #00FFFF thruster glow
    'sky_blue': 'rgb(33, 150, 243)',            # #2196F3 background
}

# ============ PBR LIGHTING ============
ROBOT_LIGHTING = dict(
    ambient=0.5,    # Base visibility
    diffuse=0.6,    # Soft shadows
    specular=1.5,   # High shine for metal/ceramic
    roughness=0.2,  # Slightly polished
    fresnel=0.3     # Rim lighting
)

LIGHT_POSITION = dict(x=2, y=2, z=10)


# ============ GEOMETRY PRIMITIVES ============

def generate_sphere(
    center: Tuple[float, float, float],
    radius: float,
    resolution: int = 30,
    scale: Tuple[float, float, float] = (1, 1, 1)
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sphere/ellipsoid vertices and faces."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution // 2)
    
    x = center[0] + radius * scale[0] * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * scale[1] * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * scale[2] * np.outer(np.ones(len(u)), np.cos(v))
    
    return x.flatten(), y.flatten(), z.flatten()


def generate_cylinder(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    radius: float,
    resolution: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate cylinder vertices and faces between two points."""
    start = np.array(start)
    end = np.array(end)
    
    # Direction vector
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    direction = direction / length
    
    # Find perpendicular vectors
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, [0, 0, 1])
    else:
        perp1 = np.cross(direction, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    
    # Generate circle points
    theta = np.linspace(0, 2 * np.pi, resolution)
    
    # Create vertices for both ends
    vertices = []
    for t in [0, 1]:
        center = start + t * (end - start)
        for angle in theta:
            point = center + radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(point)
    
    vertices = np.array(vertices)
    
    # Create faces
    faces_i, faces_j, faces_k = [], [], []
    n = resolution
    for i in range(n - 1):
        # Side faces (two triangles per quad)
        faces_i.extend([i, i + n])
        faces_j.extend([i + 1, i])
        faces_k.extend([i + n, i + n + 1])
        
        faces_i.extend([i + 1, i + n + 1])
        faces_j.extend([i + n, i + 1])
        faces_k.extend([i + n + 1, i + n])
    
    return (
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        np.array(faces_i), np.array(faces_j), np.array(faces_k)
    )


def generate_cone(
    base_center: Tuple[float, float, float],
    tip: Tuple[float, float, float],
    radius: float,
    resolution: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate cone vertices and faces."""
    base = np.array(base_center)
    tip_pt = np.array(tip)
    
    direction = tip_pt - base
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    direction = direction / length
    
    # Find perpendicular vectors
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, [0, 0, 1])
    else:
        perp1 = np.cross(direction, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    
    # Generate base circle
    theta = np.linspace(0, 2 * np.pi, resolution)
    vertices = []
    
    for angle in theta:
        point = base + radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        vertices.append(point)
    
    # Add tip
    vertices.append(tip_pt)
    vertices = np.array(vertices)
    
    # Create faces (triangles from base to tip)
    tip_idx = len(vertices) - 1
    faces_i, faces_j, faces_k = [], [], []
    
    for i in range(resolution - 1):
        faces_i.append(i)
        faces_j.append(i + 1)
        faces_k.append(tip_idx)
    
    return (
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        np.array(faces_i), np.array(faces_j), np.array(faces_k)
    )


# ============ SCI-FI DRONE CLASS ============

@dataclass
class LegConfig:
    """Configuration for a single leg."""
    hip_offset: Tuple[float, float, float]
    angle: float  # Radians, rotation around Z
    length_upper: float = 0.15
    length_lower: float = 0.12


class SciFiDrone:
    """
    Generates a Sci-Fi Scout Drone robot with:
    - Ellipsoid chassis
    - Embedded sensor eye
    - Armor plating
    - Articulated legs
    - Antennae
    - Thrusters
    """
    
    def __init__(
        self,
        chassis_radius: float = 0.2,
        chassis_scale: Tuple[float, float, float] = (1.2, 1.0, 0.6),
        num_legs: int = 4,
    ):
        self.chassis_radius = chassis_radius
        self.chassis_scale = chassis_scale
        self.num_legs = num_legs
        self.traces: List[go.Mesh3d] = []
        
    def _add_trace(
        self,
        x: np.ndarray, y: np.ndarray, z: np.ndarray,
        i: np.ndarray, j: np.ndarray, k: np.ndarray,
        color: str,
        name: str = "",
        opacity: float = 1.0
    ):
        """Add a mesh trace with standard lighting."""
        if len(x) == 0:
            return
            
        trace = go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=color,
            opacity=opacity,
            flatshading=False,
            lighting=ROBOT_LIGHTING,
            lightposition=LIGHT_POSITION,
            hoverinfo='skip',
            name=name,
            showlegend=False
        )
        self.traces.append(trace)
    
    def _add_sphere_trace(
        self,
        center: Tuple[float, float, float],
        radius: float,
        color: str,
        name: str = "",
        scale: Tuple[float, float, float] = (1, 1, 1),
        resolution: int = 30
    ):
        """Add a sphere/ellipsoid as Scatter3d surface."""
        x, y, z = generate_sphere(center, radius, resolution, scale)
        
        # For spheres, use surface plot approach
        trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=color,
                opacity=1.0
            ),
            hoverinfo='skip',
            name=name,
            showlegend=False
        )
        self.traces.append(trace)
    
    def build_chassis(self):
        """Build the main ellipsoid chassis."""
        # Main body - larger ellipsoid
        x, y, z = generate_sphere(
            center=(0, 0, 0),
            radius=self.chassis_radius,
            scale=self.chassis_scale,
            resolution=40
        )
        
        # Create mesh from surface
        self._add_sphere_trace(
            center=(0, 0, 0),
            radius=self.chassis_radius,
            color=COLORS['chassis_white'],
            name="Chassis",
            scale=self.chassis_scale,
            resolution=40
        )
        
        return self
    
    def build_sensor_eye(self):
        """Build the front sensor eye."""
        # Position at front of chassis
        eye_pos = (self.chassis_radius * self.chassis_scale[0] * 0.8, 0, 0.02)
        
        self._add_sphere_trace(
            center=eye_pos,
            radius=0.04,
            color=COLORS['sensor_dark'],
            name="Sensor Eye",
            resolution=20
        )
        
        return self
    
    def build_armor_plates(self):
        """Build side armor panels."""
        # Left panel
        left_pos = (0, -self.chassis_radius * 0.9, 0)
        self._add_sphere_trace(
            center=left_pos,
            radius=0.08,
            color=COLORS['gunmetal'],
            name="Left Armor",
            scale=(1.5, 0.3, 1.0),
            resolution=25
        )
        
        # Right panel
        right_pos = (0, self.chassis_radius * 0.9, 0)
        self._add_sphere_trace(
            center=right_pos,
            radius=0.08,
            color=COLORS['gunmetal'],
            name="Right Armor",
            scale=(1.5, 0.3, 1.0),
            resolution=25
        )
        
        return self
    
    def build_leg(self, config: LegConfig):
        """Build a single articulated leg."""
        hip = np.array(config.hip_offset)
        
        # Direction based on angle
        leg_dir = np.array([
            np.cos(config.angle),
            np.sin(config.angle),
            -0.7  # Angling down
        ])
        leg_dir = leg_dir / np.linalg.norm(leg_dir)
        
        # Upper leg
        knee = hip + leg_dir * config.length_upper
        x, y, z, i, j, k = generate_cylinder(
            tuple(hip), tuple(knee), radius=0.015
        )
        self._add_trace(x, y, z, i, j, k, COLORS['industrial_yellow'], "Upper Leg")
        
        # Knee joint (sphere)
        self._add_sphere_trace(
            center=tuple(knee),
            radius=0.02,
            color=COLORS['dark_grey'],
            name="Knee Joint"
        )
        
        # Lower leg - angling forward
        lower_dir = np.array([
            np.cos(config.angle) * 0.5,
            np.sin(config.angle) * 0.5,
            -0.85
        ])
        lower_dir = lower_dir / np.linalg.norm(lower_dir)
        foot = knee + lower_dir * config.length_lower
        
        x, y, z, i, j, k = generate_cylinder(
            tuple(knee), tuple(foot), radius=0.012
        )
        self._add_trace(x, y, z, i, j, k, COLORS['industrial_yellow'], "Lower Leg")
        
        # Foot claw (cone)
        claw_tip = foot + np.array([0, 0, -0.03])
        x, y, z, i, j, k = generate_cone(
            tuple(foot), tuple(claw_tip), radius=0.015
        )
        self._add_trace(x, y, z, i, j, k, COLORS['dark_grey'], "Foot")
        
        return self
    
    def build_all_legs(self):
        """Build all legs around the chassis."""
        angles = np.linspace(0, 2 * np.pi, self.num_legs, endpoint=False)
        
        for i, angle in enumerate(angles):
            # Offset legs slightly forward
            angle_adjusted = angle + np.pi / 4
            
            hip_x = np.cos(angle_adjusted) * self.chassis_radius * 0.8
            hip_y = np.sin(angle_adjusted) * self.chassis_radius * 0.8
            hip_z = -self.chassis_radius * self.chassis_scale[2] * 0.5
            
            config = LegConfig(
                hip_offset=(hip_x, hip_y, hip_z),
                angle=angle_adjusted
            )
            self.build_leg(config)
        
        return self
    
    def build_antennae(self):
        """Build rear antennae."""
        # Two antennae at the back
        for y_offset in [-0.06, 0.06]:
            base = (-self.chassis_radius * 0.8, y_offset, self.chassis_radius * 0.4)
            tip = (-self.chassis_radius * 0.6, y_offset, self.chassis_radius * 0.9)
            
            x, y, z, i, j, k = generate_cylinder(base, tip, radius=0.005)
            self._add_trace(x, y, z, i, j, k, COLORS['gunmetal'], "Antenna")
        
        return self
    
    def build_thrusters(self):
        """Build rear thrusters with electric blue glow."""
        # Two thrusters at the back
        for y_offset in [-0.08, 0.08]:
            base = (-self.chassis_radius * 1.0, y_offset, 0)
            tip = (-self.chassis_radius * 1.3, y_offset, 0)
            
            x, y, z, i, j, k = generate_cone(base, tip, radius=0.03)
            self._add_trace(x, y, z, i, j, k, COLORS['electric_blue'], "Thruster", opacity=0.85)
        
        return self
    
    def build(self) -> List[go.Mesh3d]:
        """Build the complete drone and return traces."""
        self.traces = []
        
        self.build_chassis()
        self.build_sensor_eye()
        self.build_armor_plates()
        self.build_all_legs()
        self.build_antennae()
        self.build_thrusters()
        
        return self.traces


# ============ MAIN FUNCTION ============

def get_robot_figure(
    background_color: str = COLORS['sky_blue'],
    show_grid: bool = False
) -> go.Figure:
    """
    Generate the complete Sci-Fi Scout Drone figure.
    
    Args:
        background_color: Scene background color
        show_grid: Whether to show grid lines
        
    Returns:
        Plotly Figure with the assembled drone
    """
    # Build the drone
    drone = SciFiDrone(
        chassis_radius=0.2,
        chassis_scale=(1.2, 1.0, 0.6),
        num_legs=4
    )
    traces = drone.build()
    
    # Create figure
    fig = go.Figure(data=traces)
    
    # Configure layout
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            bgcolor=background_color,
            xaxis=dict(
                visible=False,
                showgrid=show_grid,
                showbackground=False,
            ),
            yaxis=dict(
                visible=False,
                showgrid=show_grid,
                showbackground=False,
            ),
            zaxis=dict(
                visible=False,
                showgrid=show_grid,
                showbackground=False,
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
        ),
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    
    return fig


def get_robot_trace() -> List:
    """Get just the drone traces without a figure (for embedding)."""
    drone = SciFiDrone()
    return drone.build()


# ============ TEST ============

if __name__ == "__main__":
    fig = get_robot_figure()
    fig.show()
