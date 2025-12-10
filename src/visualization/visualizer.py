"""
Lightweight Plotly utilities for demo geometry rendering.

create_demo_mesh() generates a curved car hood surface with a localized
defect patch using vertex-level coloring.

Style: High-End Milled Aluminum / Platinum finish with PBR lighting.
"""

from typing import Tuple, Optional

import numpy as np
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
# COLOR CONSTANTS - "Studio Gunmetal" Aesthetic (High Contrast)
# ─────────────────────────────────────────────────────────────────────────────
GUNMETAL_COLOR = "#787878"        # Medium Gunmetal - contrast against white UI
DEFECT_COLOR = "#B91C1C"          # Deep Red for defects
DEFECT_OPACITY = 0.95
GRID_COLOR = "#E0E0E0"            # Subtle grey grid
BACKGROUND_COLOR = "rgba(0,0,0,0)"  # Transparent background
SHADOW_COLOR = "#000000"          # Black shadow plane


# ─────────────────────────────────────────────────────────────────────────────
# PBR LIGHTING - Studio Lighting with Strong Shadows
# Low ambient = deep shadows for volume, High specular = metallic shine
# ─────────────────────────────────────────────────────────────────────────────
PBR_LIGHTING = dict(
    ambient=0.3,        # LOW ambient for strong shadows (AO effect)
    diffuse=0.5,        # Standard diffuse
    specular=1.8,       # HIGH shine - critical for metal!
    roughness=0.1,      # Smooth machined surface
    fresnel=3.0         # Strong edge lighting for metallic look
)

# Default camera position (slightly zoomed out)
DEFAULT_CAMERA = dict(
    eye=dict(x=1.5, y=1.5, z=1.5),
    up=dict(x=0, y=0, z=1)
)



def create_demo_mesh(defect_pos: Tuple[float, float, float]) -> go.Mesh3d:
    """
    Build a curved car hood surface and color a defect patch around defect_pos.
    
    Args:
        defect_pos: (x, y, z) center of the defect patch on the surface.
    
    Returns:
        Plotly Mesh3d trace with PBR metallic lighting and per-vertex colors.
    """
    # High-resolution curved surface (100 steps for smooth appearance)
    resolution = 100
    x, y = np.meshgrid(
        np.linspace(-1.0, 1.0, resolution),
        np.linspace(-1.0, 1.0, resolution)
    )
    z = 0.2 * x**2 + 0.1 * y**3
    
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = z.ravel()
    
    # Triangulation
    n_x, n_y = x.shape
    i_idx = []
    j_idx = []
    k_idx = []
    for yi in range(n_y - 1):
        for xi in range(n_x - 1):
            v0 = yi * n_x + xi
            v1 = v0 + 1
            v2 = v0 + n_x
            v3 = v2 + 1
            i_idx.extend([v0, v0])
            j_idx.extend([v2, v1])
            k_idx.extend([v3, v3])
    
    # Vertex colors: Deep Red defects on Milled Aluminum base
    defect_vec = np.array(defect_pos)
    verts = np.vstack([x_flat, y_flat, z_flat]).T
    distances = np.linalg.norm(verts - defect_vec, axis=1)
    
    # Convert hex to RGB for vertex coloring
    gunmetal_rgb = "rgb(120, 120, 120)"  # #787878 - Gunmetal for contrast
    defect_rgb = "rgb(185, 28, 28)"       # #B91C1C
    
    vertex_colors = [
        defect_rgb if d < 0.15 else gunmetal_rgb
        for d in distances
    ]
    
    return go.Mesh3d(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        i=i_idx,
        j=j_idx,
        k=k_idx,
        vertexcolor=vertex_colors,
        flatshading=False,
        lighting=PBR_LIGHTING,
        lightposition=dict(x=0.0, y=0.0, z=2.5),
        opacity=DEFECT_OPACITY,
        name="Part Surface"
    )


def get_premium_layout(title: Optional[str] = None) -> dict:
    """
    Returns a Plotly layout configured for premium 3D visualization.
    
    Features:
    - Transparent background (blends with app)
    - Subtle grey grid lines
    - Clean, minimal axes
    
    Returns:
        dict: Layout configuration for go.Figure.update_layout()
    """
    axis_config = dict(
        showbackground=True,
        backgroundcolor=BACKGROUND_COLOR,
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        showspikes=False,
        title="",
    )
    
    return dict(
        scene=dict(
            bgcolor=BACKGROUND_COLOR,
            xaxis=axis_config,
            yaxis=axis_config,
            zaxis=axis_config,
            aspectmode="data",
            camera=DEFAULT_CAMERA
        ),
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        title=dict(
            text=title or "",
            font=dict(color="#2D2D2D", size=16)
        ),
        margin=dict(l=0, r=0, t=40 if title else 0, b=0),
        showlegend=False
    )


def create_shadow_plane(min_z: float, center: Tuple[float, float] = (0, 0), radius: float = 0.5) -> go.Mesh3d:
    """
    Create a circular shadow plane to ground objects in the scene.
    
    Args:
        min_z: Z position of the mesh bottom (shadow sits just below).
        center: (x, y) center of the shadow disc.
        radius: Radius of the shadow disc (1.5x object size recommended).
    
    Returns:
        Plotly Mesh3d trace for the shadow catcher.
    """
    # Create circular disc
    n_segments = 48  # Higher resolution for smoother edge
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    
    # Vertices: center + perimeter points
    x = [center[0]] + [center[0] + radius * np.cos(t) for t in theta]
    y = [center[1]] + [center[1] + radius * np.sin(t) for t in theta]
    z = [min_z - 0.05] * (n_segments + 1)  # 0.05 below for clear grounding
    
    # Fan triangulation from center
    i_idx = [0] * n_segments
    j_idx = list(range(1, n_segments + 1))
    k_idx = [i + 1 if i < n_segments else 1 for i in j_idx]
    
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i_idx, j=j_idx, k=k_idx,
        color=SHADOW_COLOR,
        opacity=0.15,  # Stronger shadow for grounding
        flatshading=True,
        lighting=dict(ambient=1.0, diffuse=0, specular=0),
        name="Shadow"
    )


def generate_demo_part() -> go.Figure:
    """
    Generate a complete demo 3D figure with premium styling.
    
    Returns:
        go.Figure: Ready-to-display Plotly figure with aluminum finish.
    """
    # Create mesh with sample defect position
    mesh = create_demo_mesh(defect_pos=(0.3, 0.3, 0.1))
    
    # Create shadow plane at Z=0
    shadow = create_shadow_plane(min_z=0.0, center=(0, 0), radius=0.5)
    
    # Build figure with premium layout
    fig = go.Figure(data=[shadow, mesh])  # Shadow first so mesh renders on top
    fig.update_layout(**get_premium_layout())
    
    return fig
