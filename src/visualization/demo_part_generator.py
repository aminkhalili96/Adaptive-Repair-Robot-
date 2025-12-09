
import numpy as np
import plotly.graph_objects as go

def generate_demo_part():
    """
    Generates a high-quality 3D mesh of a curved surface with a "painted" defect,
    styled with industrial-like lighting.

    Returns:
        go.Figure: A Plotly Figure object containing the configured go.Mesh3d trace.
    """
    # 1. Procedural Geometry (The Car Hood)
    grid_size = 50
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

    # 2. "Heatmap" Defect Painting
    defect_center = np.array([0.2, 0.2, -0.1 * (0.2**2 + 0.2**2)]) # Center on the surface
    defect_radius = 0.25
    
    # Default color: Metallic Silver
    vertex_colors = np.full((len(vertices), 3), 192)

    # Find vertices within the defect radius
    distances = np.linalg.norm(vertices - defect_center, axis=1)
    defect_indices = np.where(distances < defect_radius)[0]

    # Apply Rust Red color to defect vertices
    rust_red = [180, 50, 50]
    vertex_colors[defect_indices] = rust_red

    # 3. Industrial Lighting (PBR Style) and Mesh3d Trace
    mesh_trace = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i_vals,
        j=j_vals,
        k=k_vals,
        vertexcolor=vertex_colors,
        flatshading=False,
        lighting={
            'ambient': 0.3,
            'diffuse': 0.6,
            'roughness': 0.1,
            'specular': 0.5,
            'fresnel': 0.2
        },
        lightposition={'x': 100, 'y': 100, 'z': 2000}
    )

    return mesh_trace
