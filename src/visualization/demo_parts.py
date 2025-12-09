"""
Procedural Demo Part Generator.

Creates realistic industrial surfaces with vertex-colored defects
for demonstration purposes. Parts have metallic appearance with
defects "painted on" rather than floating markers.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import plotly.graph_objects as go


@dataclass
class DemoPart:
    """Container for procedurally generated demo part."""
    name: str
    vertices: np.ndarray  # (N, 3) array
    faces: np.ndarray     # (M, 3) array of vertex indices
    vertex_colors: np.ndarray  # (N,) array of color strings
    defects: List[Dict[str, Any]]
    bounds: np.ndarray


# Industrial metallic color palette
METAL_COLORS = {
    'silver': '#A8A9AD',
    'dark_silver': '#71797E',
    'steel': '#43464B',
    'aluminum': '#848789',
}

DEFECT_COLORS = {
    'rust': '#8B4513',
    'deep_rust': '#A52A2A',
    'dent': '#FF8C00',
    'scratch': '#CD853F',
    'corrosion': '#556B2F',
    'crack': '#2F4F4F',
}


def generate_curved_hood(
    width: float = 0.8,
    length: float = 1.2,
    resolution: int = 50,
    curvature: float = 0.15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a smooth curved car hood surface.
    
    Args:
        width: Hood width in meters
        length: Hood length in meters
        resolution: Grid resolution (points per axis)
        curvature: Maximum height of curve
        
    Returns:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of face indices
    """
    # Create mesh grid
    x = np.linspace(-width/2, width/2, resolution)
    y = np.linspace(-length/2, length/2, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Smooth curved surface using combination of parabola and sine
    # Main curvature (parabolic in both directions)
    Z_base = curvature * (1 - (2*X/width)**2) * (1 - (2*Y/length)**2)
    
    # Add subtle waviness for realism
    Z_wave = 0.005 * np.sin(5 * np.pi * X / width) * np.cos(3 * np.pi * Y / length)
    
    # Front edge curves down (like real hood)
    front_curve = -0.03 * np.exp(-((Y - length/2) / 0.1)**2)
    
    Z = Z_base + Z_wave + front_curve
    
    # Flatten to vertex array
    vertices = np.column_stack([
        X.flatten(),
        Y.flatten(),
        Z.flatten()
    ])
    
    # Generate triangulation (grid to triangles)
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            # Vertex indices in flattened array
            v0 = i * resolution + j
            v1 = i * resolution + j + 1
            v2 = (i + 1) * resolution + j
            v3 = (i + 1) * resolution + j + 1
            
            # Two triangles per grid cell
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    return vertices, np.array(faces)


def generate_industrial_panel(
    width: float = 0.6,
    height: float = 0.4,
    resolution: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a flat industrial panel with subtle surface imperfections.
    """
    x = np.linspace(-width/2, width/2, resolution)
    y = np.linspace(-height/2, height/2, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Mostly flat with subtle waviness
    Z = 0.002 * np.sin(10 * np.pi * X / width) * np.cos(8 * np.pi * Y / height)
    
    vertices = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v0 = i * resolution + j
            v1 = i * resolution + j + 1
            v2 = (i + 1) * resolution + j
            v3 = (i + 1) * resolution + j + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    return vertices, np.array(faces)


def apply_defect_colors(
    vertices: np.ndarray,
    defects: List[Dict[str, Any]],
    base_color: str = '#A8A9AD',
    defect_radius: float = 0.05
) -> np.ndarray:
    """
    Apply vertex colors based on defect positions.
    
    Creates a "painted on" effect where defects color nearby vertices
    instead of floating as separate markers.
    
    Args:
        vertices: (N, 3) vertex positions
        defects: List of defect dicts with 'position', 'type', 'severity'
        base_color: Default metallic color
        defect_radius: Radius of defect effect
        
    Returns:
        colors: (N,) array of hex color strings
    """
    n_vertices = len(vertices)
    
    # Initialize all vertices to base metallic color
    colors = np.full(n_vertices, base_color, dtype=object)
    
    # Color intensity array (for blending overlapping defects)
    intensities = np.zeros(n_vertices)
    
    for defect in defects:
        pos = np.array(defect['position'])
        dtype = defect.get('type', 'rust').lower()
        severity = defect.get('severity', 'medium')
        
        # Calculate distances from defect center
        distances = np.linalg.norm(vertices - pos, axis=1)
        
        # Adjust radius based on severity
        radius_multiplier = {'high': 1.5, 'medium': 1.0, 'low': 0.6}.get(severity, 1.0)
        effective_radius = defect_radius * radius_multiplier
        
        # Find affected vertices
        affected = distances < effective_radius
        
        # Get defect color based on type
        if 'rust' in dtype or 'corrosion' in dtype:
            defect_color = DEFECT_COLORS['rust']
        elif 'dent' in dtype:
            defect_color = DEFECT_COLORS['dent']
        elif 'scratch' in dtype:
            defect_color = DEFECT_COLORS['scratch']
        elif 'crack' in dtype:
            defect_color = DEFECT_COLORS['crack']
        else:
            defect_color = DEFECT_COLORS['deep_rust']
        
        # Apply color with distance-based falloff
        for idx in np.where(affected)[0]:
            # Intensity falls off with distance
            intensity = 1.0 - (distances[idx] / effective_radius)
            
            if intensity > intensities[idx]:
                colors[idx] = defect_color
                intensities[idx] = intensity
    
    return colors


def create_demo_car_hood() -> DemoPart:
    """
    Create a demo car hood with realistic defects.
    
    Returns:
        DemoPart with vertices, faces, colors, and defects
    """
    vertices, faces = generate_curved_hood(
        width=0.8,
        length=1.2,
        resolution=60,
        curvature=0.12
    )
    
    # Define realistic defects
    defects = [
        {
            'position': (0.15, 0.3, 0.08),
            'type': 'rust_spot',
            'severity': 'high',
            'confidence': 0.94,
            'normal': (0, 0, 1)
        },
        {
            'position': (-0.2, -0.2, 0.06),
            'type': 'dent',
            'severity': 'medium',
            'confidence': 0.87,
            'normal': (0, 0, 1)
        },
        {
            'position': (0.25, -0.4, 0.04),
            'type': 'scratch',
            'severity': 'low',
            'confidence': 0.78,
            'normal': (0, 0, 1)
        },
        {
            'position': (-0.1, 0.5, 0.02),
            'type': 'corrosion',
            'severity': 'high',
            'confidence': 0.91,
            'normal': (0, 0, 1)
        },
    ]
    
    # Apply defect colors to vertices
    vertex_colors = apply_defect_colors(vertices, defects, defect_radius=0.06)
    
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    
    return DemoPart(
        name="Car Hood Panel",
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        defects=defects,
        bounds=bounds
    )


def create_demo_fender() -> DemoPart:
    """Create a demo car fender with defects."""
    # Create curved surface
    resolution = 50
    theta = np.linspace(0, np.pi/2, resolution)  # Quarter cylinder
    z = np.linspace(0, 0.5, resolution)
    THETA, Z = np.meshgrid(theta, z)
    
    radius = 0.3
    X = radius * np.cos(THETA)
    Y = radius * np.sin(THETA)
    
    vertices = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v0 = i * resolution + j
            v1 = i * resolution + j + 1
            v2 = (i + 1) * resolution + j
            v3 = (i + 1) * resolution + j + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    defects = [
        {
            'position': (0.2, 0.15, 0.25),
            'type': 'dent',
            'severity': 'high',
            'confidence': 0.92,
            'normal': (0.7, 0.7, 0)
        },
        {
            'position': (0.1, 0.25, 0.4),
            'type': 'rust',
            'severity': 'medium',
            'confidence': 0.85,
            'normal': (0.5, 0.85, 0)
        },
    ]
    
    vertex_colors = apply_defect_colors(vertices, defects, defect_radius=0.05)
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    
    return DemoPart(
        name="Car Fender",
        vertices=vertices,
        faces=np.array(faces),
        vertex_colors=vertex_colors,
        defects=defects,
        bounds=bounds
    )


def render_demo_part(part: DemoPart, height: int = 600) -> go.Figure:
    """
    Render a demo part with industrial metallic lighting.
    
    Args:
        part: DemoPart instance
        height: Figure height in pixels
        
    Returns:
        Plotly Figure with metallic mesh
    """
    # Industrial lighting settings
    lighting = dict(
        ambient=0.4,
        diffuse=0.5,
        roughness=0.1,
        specular=0.4,
        fresnel=0.2
    )
    
    lightposition = dict(
        x=1000,
        y=1000,
        z=2000
    )
    
    # Create mesh with vertex colors
    mesh = go.Mesh3d(
        x=part.vertices[:, 0],
        y=part.vertices[:, 1],
        z=part.vertices[:, 2],
        i=part.faces[:, 0],
        j=part.faces[:, 1],
        k=part.faces[:, 2],
        vertexcolor=part.vertex_colors,
        lighting=lighting,
        lightposition=lightposition,
        flatshading=False,
        name=part.name,
        hoverinfo='text',
        hovertext=[f'Vertex {i}' for i in range(len(part.vertices))],
    )
    
    fig = go.Figure(data=[mesh])
    
    # Dark industrial theme
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showbackground=True,
                backgroundcolor='#1a1a1a',
                gridcolor='#333',
                showgrid=True,
                zeroline=False,
            ),
            yaxis=dict(
                showbackground=True,
                backgroundcolor='#1a1a1a',
                gridcolor='#333',
                showgrid=True,
                zeroline=False,
            ),
            zaxis=dict(
                showbackground=True,
                backgroundcolor='#1a1a1a',
                gridcolor='#333',
                showgrid=True,
                zeroline=False,
            ),
            bgcolor='#1a1a1a',
            aspectmode='data',
        ),
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        margin=dict(l=0, r=0, t=30, b=0),
        height=height,
        showlegend=False,
    )
    
    # Camera position
    camera = dict(
        eye=dict(x=1.5, y=1.5, z=1.0),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1)
    )
    fig.update_layout(scene_camera=camera)
    
    return fig


# Demo part registry
DEMO_PARTS = {
    'car_hood': ('Demo: Car Hood', create_demo_car_hood),
    'fender': ('Demo: Fender', create_demo_fender),
}


def get_demo_part_names() -> List[Tuple[str, str]]:
    """Get list of (key, display_name) for demo parts."""
    return [(k, v[0]) for k, v in DEMO_PARTS.items()]


def load_demo_part(part_key: str) -> Optional[DemoPart]:
    """Load a demo part by key."""
    if part_key in DEMO_PARTS:
        _, create_func = DEMO_PARTS[part_key]
        return create_func()
    return None
