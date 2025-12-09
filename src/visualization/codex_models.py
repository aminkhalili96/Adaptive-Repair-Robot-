"""
Industrial demo meshes with vertex-colored defects for Plotly/Streamlit.

Each generator returns a `plotly.graph_objects.Mesh3d` using flat shading and
per-vertex colors to highlight defects directly on the geometry.
"""

from typing import Tuple, List

import numpy as np
import plotly.graph_objects as go

BASE_COLOR = "rgb(200, 200, 220)"  # Light Steel Blue
DEFECT_COLOR = "rgb(255, 100, 0)"  # Bright Safety Orange


def _grid_faces(nx: int, ny: int) -> Tuple[List[int], List[int], List[int]]:
    """Create triangle indices for a regular grid (ny rows, nx cols)."""
    i_idx: List[int] = []
    j_idx: List[int] = []
    k_idx: List[int] = []
    for yi in range(ny - 1):
        for xi in range(nx - 1):
            v0 = yi * nx + xi
            v1 = v0 + 1
            v2 = v0 + nx
            v3 = v2 + 1
            i_idx.extend([v0, v0])
            j_idx.extend([v2, v1])
            k_idx.extend([v3, v3])
    return i_idx, j_idx, k_idx


def _add_box(origin: Tuple[float, float, float], size: Tuple[float, float, float]):
    """
    Build vertices and faces for an axis-aligned box.
    
    Returns:
        vertices: (8, 3)
        faces: lists i, j, k for 12 triangles
    """
    ox, oy, oz = origin
    lx, ly, lz = size
    # 8 vertices
    verts = np.array([
        [ox, oy, oz],
        [ox + lx, oy, oz],
        [ox + lx, oy + ly, oz],
        [ox, oy + ly, oz],
        [ox, oy, oz + lz],
        [ox + lx, oy, oz + lz],
        [ox + lx, oy + ly, oz + lz],
        [ox, oy + ly, oz + lz],
    ])
    # Triangles for 6 faces (two each)
    faces = [
        (0, 1, 2), (0, 2, 3),  # bottom
        (4, 5, 6), (4, 6, 7),  # top
        (0, 1, 5), (0, 5, 4),  # front
        (1, 2, 6), (1, 6, 5),  # right
        (2, 3, 7), (2, 7, 6),  # back
        (3, 0, 4), (3, 4, 7),  # left
    ]
    i_idx, j_idx, k_idx = zip(*faces)
    return verts, list(i_idx), list(j_idx), list(k_idx)


def generate_i_beam() -> go.Mesh3d:
    """
    Create a steel I-beam section with a rust patch on the web.
    Uses high-resolution grid mesh for quality rendering.
    """
    length = 1.2
    height = 0.4
    flange_width = 0.25
    flange_thickness = 0.04
    web_thickness = 0.06
    
    # Resolution
    nx = 60  # along length
    ny = 20  # along width
    
    all_verts = []
    all_i = []
    all_j = []
    all_k = []
    all_colors = []
    
    def add_surface(x_arr, y_arr, z_arr, is_web=False):
        """Add a grid surface to the mesh."""
        base_idx = len(all_verts)
        
        for yi in range(y_arr.shape[0]):
            for xi in range(y_arr.shape[1]):
                all_verts.append([x_arr[yi, xi], y_arr[yi, xi], z_arr[yi, xi]])
                
                # Color - rust patch on web
                if is_web:
                    x_pos = x_arr[yi, xi]
                    z_pos = z_arr[yi, xi]
                    # Rust in center region
                    if (0.3 < x_pos < 0.8 and 
                        height * 0.3 < z_pos < height * 0.7):
                        all_colors.append(DEFECT_COLOR)
                    else:
                        all_colors.append(BASE_COLOR)
                else:
                    all_colors.append(BASE_COLOR)
        
        # Add triangles
        rows, cols = y_arr.shape
        for yi in range(rows - 1):
            for xi in range(cols - 1):
                v0 = base_idx + yi * cols + xi
                v1 = v0 + 1
                v2 = v0 + cols
                v3 = v2 + 1
                all_i.extend([v0, v0])
                all_j.extend([v2, v1])
                all_k.extend([v3, v3])
    
    # Create coordinate grids
    x_line = np.linspace(0, length, nx)
    
    # Bottom flange - top surface
    x, y = np.meshgrid(x_line, np.linspace(-flange_width/2, flange_width/2, ny))
    z = np.full_like(x, flange_thickness)
    add_surface(x, y, z)
    
    # Bottom flange - bottom surface
    z = np.full_like(x, 0)
    add_surface(x, y, z)
    
    # Top flange - bottom surface
    z = np.full_like(x, height - flange_thickness)
    add_surface(x, y, z)
    
    # Top flange - top surface
    z = np.full_like(x, height)
    add_surface(x, y, z)
    
    # Web - front surface
    x, z = np.meshgrid(x_line, np.linspace(flange_thickness, height - flange_thickness, ny))
    y = np.full_like(x, web_thickness/2)
    add_surface(x, y, z, is_web=True)
    
    # Web - back surface (with rust)
    y = np.full_like(x, -web_thickness/2)
    add_surface(x, y, z, is_web=True)
    
    verts_arr = np.array(all_verts)
    
    return go.Mesh3d(
        x=verts_arr[:, 0],
        y=verts_arr[:, 1],
        z=verts_arr[:, 2],
        i=all_i,
        j=all_j,
        k=all_k,
        vertexcolor=all_colors,
        flatshading=True,
        lighting=dict(ambient=0.35, diffuse=0.7, specular=0.4, roughness=0.3)
    )


def generate_corrugated_sheet() -> go.Mesh3d:
    """
    Corrugated roofing sheet z = sin(x) with an orange dent in one trough.
    """
    x, y = np.meshgrid(
        np.linspace(-3.0, 3.0, 120),
        np.linspace(-1.0, 1.0, 40)
    )
    z = np.sin(x)
    
    # Apply a dent around a trough center
    dent_center = (-np.pi / 2, 0.0)
    dent_radius = 0.6
    dent_depth = 0.6
    dent_mask = ( (x - dent_center[0])**2 + (y - dent_center[1])**2 ) ** 0.5
    dent_effect = np.exp(-(dent_mask / dent_radius) ** 2)
    z = z - dent_depth * dent_effect
    
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = z.ravel()
    
    i_idx, j_idx, k_idx = _grid_faces(x.shape[1], x.shape[0])
    
    colors = [BASE_COLOR] * len(x_flat)
    defect_zone = dent_mask.ravel() < dent_radius * 0.9
    for idx, is_defect in enumerate(defect_zone):
        if is_defect:
            colors[idx] = DEFECT_COLOR
    
    return go.Mesh3d(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        i=i_idx,
        j=j_idx,
        k=k_idx,
        vertexcolor=colors,
        flatshading=True,
        lighting=dict(ambient=0.4, diffuse=0.7, specular=0.3, roughness=0.35)
    )


def generate_storage_tank_dome() -> go.Mesh3d:
    """
    Hemispherical dome with a crack-like orange line near the apex.
    """
    radius = 1.0
    theta, phi = np.meshgrid(
        np.linspace(0.0, np.pi / 2, 80),   # 0 = apex, pi/2 = equator
        np.linspace(0.0, 2 * np.pi, 120)
    )
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = z.ravel()
    
    i_idx, j_idx, k_idx = _grid_faces(x.shape[1], x.shape[0])
    
    colors = [BASE_COLOR] * len(x_flat)
    
    theta_flat = theta.ravel()
    phi_flat = phi.ravel()
    # Crack pattern near apex: narrow bands following a sine in phi
    crack_mask = (theta_flat < 0.35) & (np.abs(np.sin(6 * phi_flat)) < 0.12)
    for idx, is_crack in enumerate(crack_mask):
        if is_crack:
            colors[idx] = DEFECT_COLOR
    
    return go.Mesh3d(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        i=i_idx,
        j=j_idx,
        k=k_idx,
        vertexcolor=colors,
        flatshading=True,
        lighting=dict(ambient=0.3, diffuse=0.65, specular=0.5, roughness=0.2)
    )
