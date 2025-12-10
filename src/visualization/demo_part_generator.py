
import numpy as np
import plotly.graph_objects as go

def generate_demo_part():
    """
    Generates a high-quality 3D mesh of a curved car hood surface
    with ProAI Aesthetic styling (satin aluminum, high-res geometry).

    Returns:
        go.Mesh3d: A Plotly Mesh3d trace configured for premium industrial look.
    """
    # 1. HIGH-RES Procedural Geometry (100x100 grid for smooth curves)
    grid_size = 100  # Increased from 50 for smoother appearance
    x_range = np.linspace(-1, 1, grid_size)
    y_range = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Use a shallow parabola to mimic a curved car hood
    Z = -0.1 * (X**2 + Y**2)

    # Flatten the grid to get a list of vertices
    vertices = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Generate triangle indices (faces) for the mesh
    indices = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            p1 = i * grid_size + j
            p2 = p1 + 1
            p3 = (i + 1) * grid_size + j
            p4 = p3 + 1
            indices.append([p1, p3, p2])
            indices.append([p2, p3, p4])
    
    indices = np.array(indices)
    i_vals, j_vals, k_vals = indices[:, 0], indices[:, 1], indices[:, 2]

    # 2. ProAI Defect Painting (Alert Red on Satin Aluminum)
    defect_center = np.array([0.2, 0.2, -0.1 * (0.2**2 + 0.2**2)])
    defect_radius = 0.25
    
    # Satin Aluminum base color (#F0F2F5)
    satin_aluminum = [240, 242, 245]
    vertex_colors = np.full((len(vertices), 3), satin_aluminum)

    # Find vertices within the defect radius
    distances = np.linalg.norm(vertices - defect_center, axis=1)
    
    # Smooth gradient falloff for defect
    influence = np.clip(1.0 - (distances / defect_radius), 0, 1) ** 2
    
    # Alert Red defect color (#DC2626)
    alert_red = np.array([220, 38, 38])
    
    for i in range(len(vertices)):
        if influence[i] > 0.01:
            t = influence[i] * 0.9  # 90% opacity
            vertex_colors[i] = (
                satin_aluminum * (1 - t) + alert_red * t
            ).astype(int)

    # 3. ProAI Aesthetic Lighting (Satin Finish)
    mesh_trace = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i_vals,
        j=j_vals,
        k=k_vals,
        vertexcolor=vertex_colors,
        flatshading=False,  # Smooth Gouraud shading
        lighting={
            'ambient': 0.65,   # High ambient - no harsh shadows
            'diffuse': 0.4,    # Soft directional
            'roughness': 0.1,  # Smooth satin finish
            'specular': 0.4,   # Soft satin shine
            'fresnel': 1.0     # Rim lighting
        },
        lightposition={'x': 0, 'y': 0, 'z': 2000}
    )

    return mesh_trace

