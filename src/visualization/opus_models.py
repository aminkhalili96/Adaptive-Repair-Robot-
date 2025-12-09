"""
OPUS Industrial 3D Models.

Generates high-fidelity industrial meshes with:
- Premium metallic PBR lighting
- Vertex-colored defects (painted on, not markers)
- Proper triangulation for smooth rendering

Models:
1. Turbine Blade - Twisted airfoil with heat fracture
2. Car Fender - Compound curve with rust patch
3. Propeller Blade - Marine blade with pitting corrosion
"""

import numpy as np
import plotly.graph_objects as go
from typing import Tuple, List, Optional


# ============ COLOR PALETTE ============
METALLIC_SILVER = '#A8A9AD'
METALLIC_DARK = '#71797E'
RUST_RED = '#8B4513'
RUST_ORANGE = '#CD853F'
CORROSION_DARK = '#A52A2A'
HEAT_STRESS = '#B22222'


# ============ LIGHTING CONFIG ============
PREMIUM_LIGHTING = dict(
    ambient=0.4,
    diffuse=0.5,
    roughness=0.2,
    specular=0.6,
    fresnel=0.3
)

PREMIUM_LIGHTPOSITION = dict(
    x=1000,
    y=1000,
    z=2000
)


# ============ HELPER FUNCTIONS ============

def compute_vertex_colors(
    vertices: np.ndarray,
    defect_points: List[Tuple[float, float, float]],
    defect_radii: List[float],
    defect_colors: List[str],
    base_color: str = METALLIC_SILVER
) -> np.ndarray:
    """
    Compute vertex colors based on distance from defect points.
    
    Args:
        vertices: (N, 3) array of vertex positions
        defect_points: List of (x, y, z) defect center positions
        defect_radii: Radius of each defect effect
        defect_colors: Color for each defect
        base_color: Default metallic color
        
    Returns:
        colors: (N,) array of hex color strings
    """
    n_vertices = len(vertices)
    colors = np.full(n_vertices, base_color, dtype=object)
    intensities = np.zeros(n_vertices)
    
    for pos, radius, color in zip(defect_points, defect_radii, defect_colors):
        pos = np.array(pos)
        distances = np.linalg.norm(vertices - pos, axis=1)
        affected = distances < radius
        
        for idx in np.where(affected)[0]:
            intensity = 1.0 - (distances[idx] / radius)
            if intensity > intensities[idx]:
                colors[idx] = color
                intensities[idx] = intensity
    
    return colors


def triangulate_grid(n_rows: int, n_cols: int) -> np.ndarray:
    """
    Generate triangle indices for a grid mesh.
    
    Args:
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        
    Returns:
        faces: (M, 3) array of vertex indices
    """
    faces = []
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            v0 = i * n_cols + j
            v1 = i * n_cols + j + 1
            v2 = (i + 1) * n_cols + j
            v3 = (i + 1) * n_cols + j + 1
            
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    return np.array(faces)


# ============ TURBINE BLADE ============

def generate_turbine_blade(
    height: float = 0.4,
    chord_root: float = 0.12,
    chord_tip: float = 0.04,
    twist_angle: float = 30.0,
    n_span: int = 50,
    n_profile: int = 40
) -> go.Mesh3d:
    """
    Generate a high-fidelity twisted turbine blade.
    
    Features:
    - NACA 4-digit airfoil cross-section
    - Linear twist from root to tip
    - Tapered chord length
    - Heat-stress fracture defect on leading edge
    
    Returns:
        Plotly Mesh3d object with vertex coloring
    """
    # Generate vertices
    vertices = []
    
    for i in range(n_span):
        # Spanwise position (0 = root, 1 = tip)
        t = i / (n_span - 1)
        z = t * height
        
        # Chord varies along span (tapered)
        chord = chord_root * (1 - t) + chord_tip * t
        
        # Twist increases toward tip
        twist = np.radians(t * twist_angle)
        
        # Thickness distribution
        thickness = 0.12 * chord * (1 - 0.3 * t)
        
        # Generate NACA-style airfoil profile
        for j in range(n_profile):
            theta = 2 * np.pi * j / n_profile
            
            # Airfoil shape
            if theta <= np.pi:
                # Upper surface
                x_local = 0.5 * chord * (1 - np.cos(theta))
                x_norm = x_local / chord if chord > 0 else 0
                y_local = thickness * (
                    0.2969 * np.sqrt(max(x_norm, 0)) 
                    - 0.1260 * x_norm 
                    - 0.3516 * x_norm**2 
                    + 0.2843 * x_norm**3 
                    - 0.1015 * x_norm**4
                ) * 5
            else:
                # Lower surface
                x_local = 0.5 * chord * (1 - np.cos(2*np.pi - theta))
                x_norm = x_local / chord if chord > 0 else 0
                y_local = -thickness * (
                    0.2969 * np.sqrt(max(x_norm, 0)) 
                    - 0.1260 * x_norm 
                    - 0.3516 * x_norm**2 
                    + 0.2843 * x_norm**3 
                    - 0.1015 * x_norm**4
                ) * 5
            
            # Apply twist rotation
            x_twisted = (x_local - chord/2) * np.cos(twist) - y_local * np.sin(twist)
            y_twisted = (x_local - chord/2) * np.sin(twist) + y_local * np.cos(twist)
            
            vertices.append([x_twisted, y_twisted, z])
    
    vertices = np.array(vertices)
    
    # Generate faces
    faces = triangulate_grid(n_span, n_profile)
    
    # Close the profile (wrap around)
    for i in range(n_span - 1):
        v0 = i * n_profile + (n_profile - 1)
        v1 = i * n_profile
        v2 = (i + 1) * n_profile + (n_profile - 1)
        v3 = (i + 1) * n_profile
        faces = np.vstack([faces, [v0, v1, v2], [v1, v3, v2]])
    
    # Define heat-stress fracture defect on leading edge
    defect_points = [
        (0.03, 0.01, height * 0.65),   # Primary fracture
        (0.02, 0.005, height * 0.55),  # Secondary damage
    ]
    defect_radii = [0.025, 0.015]
    defect_colors = [HEAT_STRESS, RUST_RED]
    
    # Compute vertex colors
    colors = compute_vertex_colors(
        vertices, defect_points, defect_radii, defect_colors
    )
    
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=colors,
        lighting=PREMIUM_LIGHTING,
        lightposition=PREMIUM_LIGHTPOSITION,
        flatshading=False,
        name="Turbine Blade",
        hoverinfo='name'
    )


# ============ CAR FENDER ============

def generate_car_fender(
    width: float = 0.6,
    height: float = 0.25,
    depth: float = 0.4,
    n_width: int = 40,
    n_depth: int = 50
) -> go.Mesh3d:
    """
    Generate a smooth compound-curved car fender.
    
    Features:
    - Double parabolic curvature (width and depth)
    - Smooth edge roll at top
    - Rust patch defect on top arch
    
    Returns:
        Plotly Mesh3d object with vertex coloring
    """
    # Create parametric surface
    u = np.linspace(0, 1, n_width)
    v = np.linspace(0, 1, n_depth)
    U, V = np.meshgrid(u, v)
    
    # X: Width with parabolic arch
    X = (U - 0.5) * width
    
    # Y: Depth (fender protrusion)
    Y = V * depth
    
    # Z: Height with compound curvature
    # Main arch in width direction
    arch_width = height * 4 * U * (1 - U)
    # Secondary curve in depth direction (fender roll)
    arch_depth = 0.3 * height * np.sin(np.pi * V)
    # Edge roll at top
    edge_roll = 0.05 * height * np.exp(-((U - 0.5) / 0.1)**2) * (1 - V)
    
    Z = arch_width + arch_depth + edge_roll
    
    # Flatten to vertex array
    vertices = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    # Generate faces
    faces = triangulate_grid(n_depth, n_width)
    
    # Define rust patch defect on top of arch
    defect_points = [
        (0.0, depth * 0.3, height * 0.9),     # Main rust spot
        (0.08, depth * 0.4, height * 0.85),   # Secondary rust
        (-0.05, depth * 0.25, height * 0.88), # Spreading rust
    ]
    defect_radii = [0.06, 0.035, 0.025]
    defect_colors = [RUST_ORANGE, RUST_RED, CORROSION_DARK]
    
    # Compute vertex colors
    colors = compute_vertex_colors(
        vertices, defect_points, defect_radii, defect_colors
    )
    
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=colors,
        lighting=PREMIUM_LIGHTING,
        lightposition=PREMIUM_LIGHTPOSITION,
        flatshading=False,
        name="Car Fender",
        hoverinfo='name'
    )


# ============ PROPELLER BLADE ============

def generate_propeller_blade(
    radius: float = 0.35,
    chord_root: float = 0.15,
    chord_tip: float = 0.05,
    pitch_angle: float = 25.0,
    n_radial: int = 45,
    n_chord: int = 30
) -> go.Mesh3d:
    """
    Generate a marine propeller blade.
    
    Features:
    - Helical pitch angle
    - Tapered chord
    - Curved blade surface
    - Pitting corrosion defects
    
    Returns:
        Plotly Mesh3d object with vertex coloring
    """
    vertices = []
    
    for i in range(n_radial):
        # Radial position (0 = hub, 1 = tip)
        r_frac = i / (n_radial - 1)
        r = 0.05 + r_frac * (radius - 0.05)  # Start from hub
        
        # Chord varies along radius
        chord = chord_root * (1 - r_frac) + chord_tip * r_frac
        
        # Pitch angle (twist)
        pitch = np.radians(pitch_angle * (1 - 0.4 * r_frac))
        
        # Blade rake (slight backward curve)
        rake = 0.02 * r_frac**2
        
        for j in range(n_chord):
            # Chordwise position
            c_frac = j / (n_chord - 1)
            c = (c_frac - 0.5) * chord
            
            # Blade section shape (NACA-like thickness)
            t_max = 0.08 * chord * (1 - 0.5 * r_frac)
            thickness = t_max * (1 - 4 * (c_frac - 0.3)**2) if 0.1 < c_frac < 0.5 else t_max * 0.2
            thickness = max(thickness, 0)
            
            # Apply pitch rotation
            x = r * np.cos(c / r if r > 0 else 0) + rake
            y = r * np.sin(c / r if r > 0 else 0)
            z = c * np.sin(pitch) + thickness * np.cos(pitch)
            
            vertices.append([x, y, z])
    
    vertices = np.array(vertices)
    
    # Generate faces
    faces = triangulate_grid(n_radial, n_chord)
    
    # Define pitting corrosion defects (multiple small spots)
    defect_points = [
        (0.15, 0.08, 0.01),
        (0.20, 0.12, -0.01),
        (0.18, 0.05, 0.02),
        (0.25, 0.15, 0.00),
        (0.12, 0.10, -0.02),
        (0.22, 0.08, 0.01),
    ]
    defect_radii = [0.02, 0.015, 0.018, 0.012, 0.02, 0.015]
    defect_colors = [RUST_RED, CORROSION_DARK, RUST_RED, RUST_ORANGE, CORROSION_DARK, RUST_RED]
    
    # Compute vertex colors
    colors = compute_vertex_colors(
        vertices, defect_points, defect_radii, defect_colors
    )
    
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=colors,
        lighting=PREMIUM_LIGHTING,
        lightposition=PREMIUM_LIGHTPOSITION,
        flatshading=False,
        name="Propeller Blade",
        hoverinfo='name'
    )


# ============ FACTORY FUNCTIONS ============

def get_opus_models() -> dict:
    """
    Get all OPUS models as a dictionary.
    
    Returns:
        Dict mapping model name to (display_name, generator_function)
    """
    return {
        'turbine_blade': ('Turbine Blade (OPUS)', generate_turbine_blade),
        'car_fender': ('Car Fender (OPUS)', generate_car_fender),
        'propeller_blade': ('Propeller Blade (OPUS)', generate_propeller_blade),
    }


def create_opus_figure(mesh: go.Mesh3d, height: int = 600) -> go.Figure:
    """
    Create a Plotly figure with industrial theme for an OPUS mesh.
    
    Args:
        mesh: Mesh3d object
        height: Figure height
        
    Returns:
        Configured Plotly Figure
    """
    fig = go.Figure(data=[mesh])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showbackground=True,
                backgroundcolor='#1a1a1a',
                gridcolor='#333',
                showgrid=True,
                zeroline=False,
                title=''
            ),
            yaxis=dict(
                showbackground=True,
                backgroundcolor='#1a1a1a',
                gridcolor='#333',
                showgrid=True,
                zeroline=False,
                title=''
            ),
            zaxis=dict(
                showbackground=True,
                backgroundcolor='#1a1a1a',
                gridcolor='#333',
                showgrid=True,
                zeroline=False,
                title=''
            ),
            bgcolor='#1a1a1a',
            aspectmode='data',
        ),
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        margin=dict(l=0, r=0, t=30, b=0),
        height=height,
        showlegend=False,
        scene_camera=dict(
            eye=dict(x=1.8, y=1.8, z=1.2),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        )
    )
    
    return fig


# ============ DEFECT INFO ============

def get_opus_defects(model_name: str) -> List[dict]:
    """
    Get defect information for an OPUS model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        List of defect dicts
    """
    defects_map = {
        'turbine_blade': [
            {'position': (0.03, 0.01, 0.26), 'type': 'heat_stress_fracture', 
             'severity': 'high', 'confidence': 0.94, 'normal': (0.8, 0.4, 0.2)},
            {'position': (0.02, 0.005, 0.22), 'type': 'thermal_damage', 
             'severity': 'medium', 'confidence': 0.87, 'normal': (0.7, 0.3, 0.3)},
        ],
        'car_fender': [
            {'position': (0.0, 0.12, 0.22), 'type': 'rust_patch', 
             'severity': 'high', 'confidence': 0.92, 'normal': (0, 0.3, 0.95)},
            {'position': (0.08, 0.16, 0.21), 'type': 'surface_corrosion', 
             'severity': 'medium', 'confidence': 0.85, 'normal': (0.2, 0.4, 0.9)},
        ],
        'propeller_blade': [
            {'position': (0.15, 0.08, 0.01), 'type': 'pitting_corrosion', 
             'severity': 'high', 'confidence': 0.91, 'normal': (0.6, 0.6, 0.5)},
            {'position': (0.20, 0.12, -0.01), 'type': 'cavitation_damage', 
             'severity': 'medium', 'confidence': 0.83, 'normal': (0.5, 0.7, 0.5)},
            {'position': (0.25, 0.15, 0.0), 'type': 'pitting_corrosion', 
             'severity': 'low', 'confidence': 0.76, 'normal': (0.4, 0.8, 0.4)},
        ],
    }
    return defects_map.get(model_name, [])


if __name__ == '__main__':
    # Test generation
    print("Generating OPUS models...")
    
    blade = generate_turbine_blade()
    print(f"✓ Turbine Blade: {len(blade.x)} vertices")
    
    fender = generate_car_fender()
    print(f"✓ Car Fender: {len(fender.x)} vertices")
    
    prop = generate_propeller_blade()
    print(f"✓ Propeller Blade: {len(prop.x)} vertices")
    
    print("\nAll models generated successfully!")
