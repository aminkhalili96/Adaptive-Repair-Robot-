"""
Lightweight Plotly utilities for demo geometry rendering.

create_demo_mesh() generates a curved car hood surface with a localized
rust patch using vertex-level coloring.
"""

from typing import Tuple

import numpy as np
import plotly.graph_objects as go


def create_demo_mesh(defect_pos: Tuple[float, float, float]) -> go.Mesh3d:
    """
    Build a curved car hood surface and color a rust patch around defect_pos.
    
    Args:
        defect_pos: (x, y, z) center of the rust patch on the surface.
    
    Returns:
        Plotly Mesh3d trace with metallic lighting and per-vertex colors.
    """
    # Curved surface
    x, y = np.meshgrid(
        np.linspace(-1.0, 1.0, 60),
        np.linspace(-1.0, 1.0, 60)
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
    
    # Vertex colors based on rust proximity
    defect_vec = np.array(defect_pos)
    verts = np.vstack([x_flat, y_flat, z_flat]).T
    distances = np.linalg.norm(verts - defect_vec, axis=1)
    vertex_colors = [
        "rgb(200, 50, 50)" if d < 0.15 else "rgb(192, 192, 192)"
        for d in distances
    ]
    
    lighting = dict(
        ambient=0.25,
        diffuse=0.6,
        specular=1.0,
        roughness=0.05,
        fresnel=0.3
    )
    
    return go.Mesh3d(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        i=i_idx,
        j=j_idx,
        k=k_idx,
        vertexcolor=vertex_colors,
        flatshading=False,
        lighting=lighting,
        lightposition=dict(x=0.0, y=0.0, z=2.5)
    )
