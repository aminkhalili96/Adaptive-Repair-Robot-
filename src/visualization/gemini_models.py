
import numpy as np
import plotly.graph_objects as go

# --- Common Configuration for a Premium "Showcase" Look ---

# Lighting configuration for a bright, clear, brushed aluminum appearance
PREMIUM_LIGHTING = {
    'ambient': 0.4,
    'diffuse': 0.8,
    'specular': 0.8,
    'roughness': 0.1,
    'fresnel': 0.2
}

# Consistent light position to simulate overhead workshop lighting
PREMIUM_LIGHT_POSITION = {'x': 1000, 'y': 1000, 'z': 5000}

# --- Utility Function to create mesh grid and triangles ---

def _create_grid_and_triangles(u_res, v_res):
    """
    Generates triangle indices for a standard u,v grid.
    Returns flattened i, j, k arrays for go.Mesh3d.
    """
    indices = []
    for i in range(u_res - 1):
        for j in range(v_res - 1):
            p1 = i * v_res + j
            p2 = p1 + 1
            p3 = (i + 1) * v_res + j
            p4 = p3 + 1
            indices.append([p1, p3, p2])
            indices.append([p2, p3, p4])
    
    indices = np.array(indices)
    return indices[:, 0], indices[:, 1], indices[:, 2]

# --- Showcase Model 1: Aircraft Fuselage ---

def generate_aircraft_fuselage(resolution=100):
    """
    Generates a high-resolution, gently curved cylindrical section representing
    an aircraft fuselage, with a cluster of corrosion "rash" spots.

    Returns:
        go.Mesh3d: A Plotly trace object ready for rendering.
    """
    # 1. Geometry: Large, gently curved cylinder section
    radius = 5.0
    length = 2.5
    angle_range = np.pi / 2 # 90-degree section

    z_coords = np.linspace(-length / 2, length / 2, resolution)
    theta = np.linspace(-angle_range / 2, angle_range / 2, resolution)
    theta, z_coords = np.meshgrid(theta, z_coords)

    x_coords = radius * np.cos(theta)
    y_coords = radius * np.sin(theta)
    
    vertices = np.vstack([x_coords.ravel(), y_coords.ravel(), z_coords.ravel()]).T

    # 2. Defect: A cluster of corrosion spots in the center
    defect_centers = [
        np.array([radius, 0, 0]),
        np.array([radius * np.cos(0.05), radius * np.sin(0.05), 0.2]),
        np.array([radius * np.cos(-0.04), radius * np.sin(-0.04), -0.1]),
        np.array([radius * np.cos(0.02), radius * np.sin(0.02), 0.4]),
        np.array([radius * np.cos(-0.06), radius * np.sin(-0.06), -0.3]),
    ]
    defect_radius = 0.15
    
    # Default Color: Brushed Aluminum
    vertex_colors = np.full((len(vertices), 3), 200)

    # Color defect areas with a dark red/brown for corrosion
    corrosion_color = [139, 69, 19]
    for center in defect_centers:
        distances = np.linalg.norm(vertices - center, axis=1)
        vertex_colors[distances < defect_radius] = corrosion_color

    # 3. Mesh Construction
    i, j, k = _create_grid_and_triangles(resolution, resolution)

    fuselage_trace = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=i, j=j, k=k,
        vertexcolor=vertex_colors,
        flatshading=False,
        lighting=PREMIUM_LIGHTING,
        lightposition=PREMIUM_LIGHT_POSITION,
        name='Aircraft Fuselage'
    )
    return fuselage_trace

# --- Showcase Model 2: Complex Pipe Bend ---

def generate_complex_pipe_bend(resolution=80):
    """
    Generates a 90-degree pipe elbow (torus segment) with a rust ring
    around a simulated joint area.

    Returns:
        go.Mesh3d: A Plotly trace object ready for rendering.
    """
    # 1. Geometry: Torus segment (90-degree bend)
    major_radius = 1.0  # Radius of the bend
    minor_radius = 0.2  # Radius of the pipe itself
    
    u = np.linspace(0, np.pi / 2, resolution)  # Angle of the bend (0 to 90 degrees)
    v = np.linspace(0, 2 * np.pi, resolution)  # Angle around the pipe's cross-section
    u, v = np.meshgrid(u, v)

    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = minor_radius * np.sin(v)

    vertices = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # 2. Defect: A rust ring around a joint
    rust_color = [180, 50, 50]
    joint_position_start = np.pi / 2 - 0.2
    joint_position_end = np.pi / 2 + 0.2
    
    # Default Color: Brushed Aluminum
    vertex_colors = np.full((len(vertices), 3), 200)

    # Find vertices in the "joint" area by checking the cross-section angle `v`
    # We use the raveled `v` array which corresponds to the flattened vertex list
    joint_indices = np.where(
        (v.ravel() > joint_position_start) & (v.ravel() < joint_position_end)
    )
    vertex_colors[joint_indices] = rust_color
    
    # 3. Mesh Construction
    i, j, k = _create_grid_and_triangles(resolution, resolution)

    pipe_trace = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=i, j=j, k=k,
        vertexcolor=vertex_colors,
        flatshading=False,
        lighting=PREMIUM_LIGHTING,
        lightposition=PREMIUM_LIGHT_POSITION,
        name='Complex Pipe Bend'
    )
    return pipe_trace

# --- Showcase Model 3: Hyperbolic Paraboloid (Saddle) ---

def generate_saddle_shape(resolution=80):
    """
    Generates a hyperbolic paraboloid (Pringle/saddle shape) with a
    clear impact mark in the center depression.

    Returns:
        go.Mesh3d: A Plotly trace object ready for rendering.
    """
    # 1. Geometry: Hyperbolic Paraboloid
    x_range = np.linspace(-1, 1, resolution)
    y_range = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x_range, y_range)
    Z = 0.3 * (X**2 - Y**2)  # Scale to control curvature

    vertices = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # 2. Defect: A red impact mark in the deepest curve
    defect_center = np.array([0, 0, 0])
    defect_radius = 0.18
    impact_color = [220, 0, 0] # Bright red for a clear impact

    # Default Color: Brushed Aluminum
    vertex_colors = np.full((len(vertices), 3), 200)

    # Find vertices within the impact radius
    distances = np.linalg.norm(vertices - defect_center, axis=1)
    vertex_colors[distances < defect_radius] = impact_color

    # 3. Mesh Construction
    i, j, k = _create_grid_and_triangles(resolution, resolution)

    saddle_trace = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=i, j=j, k=k,
        vertexcolor=vertex_colors,
        flatshading=False,
        lighting=PREMIUM_LIGHTING,
        lightposition=PREMIUM_LIGHT_POSITION,
        name='Saddle Shape'
    )
    return saddle_trace
