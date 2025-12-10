"""
Industrial 3D Visualization Pipeline

Professional PBR-style rendering with:
- "Industrial Metal" material (proper lighting for steel/aluminum)
- Vertex-colored defects ("painted on" the mesh, not floating)
- Transparent background for seamless UI integration
- Proper camera positioning
"""

import plotly.graph_objects as go
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from src.visualization.mesh_loader import MeshData


# ============ PROAI AESTHETIC MATERIAL ============
# Studio Lighting with Strong Shadows for Contrast
# Lower ambient = darker shadows, higher specular = metallic shine
METAL_LIGHTING = dict(
    ambient=0.35,      # LOW ambient for strong shadows
    diffuse=0.5,       # Good directional lighting
    roughness=0.1,     # Smooth machined surface
    specular=1.8,      # HIGH shine for metal
    fresnel=3.0        # Strong rim lighting to separate from background
)

# Overhead studio lighting
LIGHT_POSITION = dict(x=0, y=0, z=10)

# ProAI Color Palette (High Contrast Industrial)
COLORS = {
    # Darker base for contrast against white UI
    'satin_aluminum': 'rgb(140, 145, 150)',   # Medium gunmetal
    'light_steel': 'rgb(140, 145, 150)',      # Medium gunmetal
    'steel_grey': 'rgb(120, 120, 120)',       # Darker gunmetal #787878
    'gunmetal': 'rgb(100, 100, 100)',         # Dark accent
    
    # Defects - Alert Red for contrast on dark surface
    'defect_rust': 'rgb(239, 68, 68)',        # Brighter red (#EF4444)
    'defect_high': 'rgb(220, 38, 38)',        # Alert Red
    'defect_medium': 'rgb(251, 146, 60)',     # Bright Orange
    'defect_low': 'rgb(34, 197, 94)',         # Bright green
    
    # Accents
    'toolpath': 'rgb(59, 130, 246)',          # Bright blue
    'highlight': 'rgb(250, 204, 21)',         # Amber
    
    # Scene
    'grid': 'rgb(229, 229, 229)',             # Subtle grid #E5E5E5
    'bg_transparent': 'rgba(0, 0, 0, 0)',     # Transparent
}


def generate_reflection_gradient(
    vertices: np.ndarray,
    base_color: str = 'rgb(240, 242, 245)',    # Satin aluminum
    highlight_tint: str = 'rgb(255, 255, 255)', # Pure white highlight
    gradient_axis: int = 2  # Z-axis for top-down reflection
) -> List[str]:
    """
    Generate subtle reflection gradient for ProAI Aesthetic.
    
    Creates a soft white-to-aluminum gradient that simulates
    soft studio lighting on satin aluminum surface.
    
    Args:
        vertices: Nx3 vertex positions
        base_color: Satin aluminum base
        highlight_tint: White highlight for top surfaces
        gradient_axis: Axis for gradient (0=X, 1=Y, 2=Z)
        
    Returns:
        List of RGB color strings per vertex
    """
    n_verts = len(vertices)
    if n_verts == 0:
        return []
    
    # Get position along gradient axis
    positions = vertices[:, gradient_axis]
    min_pos, max_pos = positions.min(), positions.max()
    
    if max_pos - min_pos < 1e-6:
        return [base_color] * n_verts
    
    # Normalize to 0-1
    t = (positions - min_pos) / (max_pos - min_pos)
    
    # Parse colors
    base_rgb = _parse_rgb(base_color)
    highlight_rgb = _parse_rgb(highlight_tint)
    
    colors = []
    for i in range(n_verts):
        # Interpolate: bottom = base, top = subtle white highlight
        factor = t[i] * 0.15  # Very subtle effect (15% max)
        r = int(base_rgb[0] * (1 - factor) + highlight_rgb[0] * factor)
        g = int(base_rgb[1] * (1 - factor) + highlight_rgb[1] * factor)
        b = int(base_rgb[2] * (1 - factor) + highlight_rgb[2] * factor)
        colors.append(f'rgb({r}, {g}, {b})')
    
    return colors


def generate_vertex_colors(
    vertices: np.ndarray,
    defects: List[Dict],
    base_color: str = 'rgb(140, 145, 150)',   # Medium gunmetal for contrast
    defect_color: str = 'rgb(239, 68, 68)',   # Bright red
    defect_radius: float = 0.03,
    defect_opacity: float = 0.9              # High visibility
) -> List[str]:
    """
    Generate vertex colors with defects "painted" onto the mesh.
    
    This creates a smooth defect heatmap by interpolating colors
    based on distance from defect centers.
    
    Args:
        vertices: Nx3 array of vertex positions
        defects: List of defect dicts with 'position' key
        base_color: Default steel grey color
        defect_color: Color for defect regions
        defect_radius: Radius of defect coloring
        
    Returns:
        List of color strings for each vertex
    """
    n_vertices = len(vertices)
    
    # Initialize all vertices to base steel color
    colors = [base_color] * n_vertices
    
    if not defects:
        return colors
    
    # Parse base RGB values
    base_rgb = _parse_rgb(base_color)
    defect_rgb = _parse_rgb(defect_color)
    
    # Calculate defect influence on each vertex
    influence = np.zeros(n_vertices)
    
    for defect in defects:
        if 'position' not in defect:
            continue
            
        defect_pos = np.array(defect['position'])
        
        # Get severity-based color if available
        severity = defect.get('severity', 'medium')
        if severity == 'high':
            defect_rgb = _parse_rgb(COLORS['defect_high'])
        elif severity == 'low':
            defect_rgb = _parse_rgb(COLORS['defect_low'])
        else:
            defect_rgb = _parse_rgb(COLORS['defect_rust'])
        
        # Calculate distance from each vertex to this defect
        distances = np.linalg.norm(vertices - defect_pos, axis=1)
        
        # Smooth falloff within radius (1 at center, 0 at edge)
        local_influence = np.clip(1.0 - (distances / defect_radius), 0, 1)
        
        # Quadratic falloff for smoother blend
        local_influence = local_influence ** 2
        
        # Update vertex colors based on influence
        for i in range(n_vertices):
            if local_influence[i] > 0.01:  # Skip negligible influence
                t = local_influence[i]
                # Interpolate between base and defect color
                r = int(base_rgb[0] * (1 - t) + defect_rgb[0] * t)
                g = int(base_rgb[1] * (1 - t) + defect_rgb[1] * t)
                b = int(base_rgb[2] * (1 - t) + defect_rgb[2] * t)
                colors[i] = f'rgb({r}, {g}, {b})'
    
    return colors


def _parse_rgb(color_str: str) -> Tuple[int, int, int]:
    """Parse 'rgb(r, g, b)' string to tuple."""
    try:
        parts = color_str.replace('rgb(', '').replace(')', '').split(',')
        return (int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip()))
    except:
        return (112, 112, 112)  # Default gunmetal


def create_ground_shadow(
    mesh_data,
    shadow_color: str = 'rgba(0, 0, 0, 0.15)',
    offset_z: float = -0.02,
    scale: float = 1.2
) -> go.Mesh3d:
    """
    Create a fake ground shadow plane beneath the mesh.
    
    This creates a visual grounding effect, making the object
    appear to sit in a physical space rather than floating.
    
    Args:
        mesh_data: MeshData object for bounds calculation
        shadow_color: RGBA color for shadow (semi-transparent black)
        offset_z: Z offset below mesh minimum
        scale: Scale factor for shadow size
        
    Returns:
        Plotly Mesh3d trace for shadow plane
    """
    bounds = mesh_data.bounds
    center = (bounds[0] + bounds[1]) / 2
    size = bounds[1] - bounds[0]
    
    # Shadow at bottom of mesh
    z_pos = bounds[0][2] + offset_z
    
    # Create elliptical shadow (oval plane)
    n_points = 32
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = center[0] + (size[0] * scale / 2) * np.cos(theta)
    y = center[1] + (size[1] * scale / 2) * np.sin(theta)
    z = np.full_like(x, z_pos)
    
    # Add center point for fan triangulation
    x = np.append(x, center[0])
    y = np.append(y, center[1])
    z = np.append(z, z_pos)
    
    # Create triangles (fan from center)
    center_idx = n_points
    i_vals = [center_idx] * n_points
    j_vals = list(range(n_points))
    k_vals = [(i + 1) % n_points for i in range(n_points)]
    
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i_vals, j=j_vals, k=k_vals,
        color=shadow_color,
        opacity=0.15,
        flatshading=True,
        hoverinfo='skip',
        showlegend=False,
        name='Shadow'
    )

def create_industrial_mesh_trace(
    mesh_data: MeshData,
    defects: Optional[List[Dict]] = None,
    defect_radius: float = 0.03,
    show_edges: bool = False
) -> go.Mesh3d:
    """
    Create a Mesh3d trace with Industrial Metal rendering.
    
    Features:
    - Proper PBR-style lighting
    - Vertex-colored defects painted on surface
    - No floating markers or cones
    
    Args:
        mesh_data: MeshData object with vertices/faces
        defects: Optional list of defect dicts with 'position'
        defect_radius: Radius for defect color bleeding
        show_edges: Whether to show wireframe edges
        
    Returns:
        Plotly Mesh3d trace
    """
    vertices = mesh_data.vertices
    faces = mesh_data.faces
    
    # Generate vertex colors with defects painted on
    vertex_colors = generate_vertex_colors(
        vertices, 
        defects or [],
        defect_radius=defect_radius
    )
    
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=vertex_colors,
        opacity=1.0,
        flatshading=False,  # Smooth shading for metal look
        lighting=METAL_LIGHTING,
        lightposition=LIGHT_POSITION,
        hoverinfo='skip',
        name='Part',
        showlegend=False
    )


def create_industrial_layout(
    mesh_data: MeshData,
    transparent_bg: bool = True
) -> Dict[str, Any]:
    """
    Create Plotly layout for ProAI Aesthetic visualization.
    
    Features:
    - Transparent background (blends with light UI)
    - Hidden axes (clean, minimal look)
    - aspectmode='data' (no stretching)
    - Proper camera positioning
    
    Args:
        mesh_data: MeshData for bounds calculation
        transparent_bg: Use transparent background (default True)
        
    Returns:
        Layout dict for go.Figure
    """
    # Calculate camera position to fill the screen
    bounds = mesh_data.bounds
    center = (bounds[0] + bounds[1]) / 2
    size = np.max(bounds[1] - bounds[0])
    
    # Camera distance to fill view
    distance = size * 2.0
    
    # ProAI: default to transparent for light UI integration
    bg_color = 'rgba(0,0,0,0)' if transparent_bg else COLORS.get('bg_transparent', 'rgba(0,0,0,0)')
    grid_color = COLORS.get('grid', 'rgb(229, 229, 229)')
    
    return dict(
        scene=dict(
            xaxis=dict(
                visible=False,
                showbackground=False,
                gridcolor=grid_color,
                zerolinecolor=grid_color,
            ),
            yaxis=dict(
                visible=False,
                showbackground=False,
                gridcolor=grid_color,
                zerolinecolor=grid_color,
            ),
            zaxis=dict(
                visible=False,
                showbackground=False,
                gridcolor=grid_color,
                zerolinecolor=grid_color,
            ),
            aspectmode='data',
            bgcolor=bg_color,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        uirevision='proai',  # Preserve camera on rerender
    )


class Mesh3DViewer:
    """
    Industrial 3D Mesh Viewer.
    
    Renders meshes with PBR-style "Industrial Metal" material
    and vertex-colored defects (painted on, not floating).
    """
    
    def __init__(self, mesh_data: MeshData):
        """Initialize with mesh data."""
        self.mesh_data = mesh_data
        self.fig = None
        self._defects: List[Dict] = []
        self._defect_radius = 0.03
        self._toolpath: List[Tuple[float, float, float]] = []
        self._highlight_position: Optional[Tuple[float, float, float]] = None
        self._highlight_radius: float = 0.02
        
    def add_defect_markers(
        self, 
        positions: List[Tuple[float, float, float]],
        labels: Optional[List[str]] = None,
        severities: Optional[List[str]] = None,
        normals: Optional[List[Tuple[float, float, float]]] = None,
        confidences: Optional[List[float]] = None
    ):
        """
        Add defects to be painted onto the mesh.
        
        Note: These are NOT floating markers - they become vertex colors.
        
        Args:
            positions: Defect center positions
            labels: Optional labels (for data)
            severities: 'high', 'medium', or 'low'
            normals: Surface normals (for toolpath)
            confidences: Detection confidence values
        """
        for i, pos in enumerate(positions):
            self._defects.append({
                'position': pos,
                'label': labels[i] if labels and i < len(labels) else f'Defect {i+1}',
                'severity': severities[i] if severities and i < len(severities) else 'medium',
                'normal': normals[i] if normals and i < len(normals) else None,
                'confidence': confidences[i] if confidences and i < len(confidences) else 0.9,
            })
    
    def add_toolpath(self, waypoints: List[Tuple[float, float, float]]):
        """Add repair toolpath visualization."""
        self._toolpath = waypoints
    
    def highlight_region(self, center: Tuple[float, float, float], radius: float = 0.02):
        """Highlight a region (adds extra defect coloring)."""
        self._highlight_position = center
        self._highlight_radius = radius
    
    def create_figure(self) -> go.Figure:
        """
        Create the industrial-styled Plotly figure.
        
        Returns:
            Plotly Figure with mesh and defects
        """
        # Create main mesh trace with vertex-colored defects
        mesh_trace = create_industrial_mesh_trace(
            self.mesh_data,
            defects=self._defects,
            defect_radius=self._defect_radius
        )
        
        traces = [mesh_trace]
        
        # Add toolpath if present (this IS a line trace)
        if self._toolpath:
            toolpath_trace = go.Scatter3d(
                x=[w[0] for w in self._toolpath],
                y=[w[1] for w in self._toolpath],
                z=[w[2] for w in self._toolpath],
                mode='lines',
                line=dict(color=COLORS['toolpath'], width=4),
                name='Toolpath',
                showlegend=False,
                hoverinfo='skip'
            )
            traces.append(toolpath_trace)
        
        # Create figure
        self.fig = go.Figure(data=traces)
        
        # Apply industrial layout
        layout = create_industrial_layout(self.mesh_data, transparent_bg=True)
        self.fig.update_layout(**layout)
        
        return self.fig
    
    def set_camera_view(self, target: Tuple[float, float, float]) -> Dict:
        """
        Calculate camera to focus on a target position.
        
        Args:
            target: Position to focus on
            
        Returns:
            Camera dict for Plotly layout
        """
        target = np.array(target)
        bounds = self.mesh_data.bounds
        size = np.max(bounds[1] - bounds[0])
        distance = size * 1.2
        
        # Camera positioned looking at target from 45 degrees
        eye = target + np.array([distance * 0.6, distance * 0.6, distance * 0.4])
        
        return dict(
            eye=dict(x=eye[0], y=eye[1], z=eye[2]),
            center=dict(x=target[0], y=target[1], z=target[2]),
            up=dict(x=0, y=0, z=1)
        )


# ============ CONVENIENCE FUNCTIONS ============

def render_industrial_mesh(
    mesh_data: MeshData,
    defects: Optional[List[Dict]] = None,
    height: int = 600
) -> go.Figure:
    """
    One-liner to render a mesh with industrial styling.
    
    Args:
        mesh_data: MeshData object
        defects: Optional defect list
        height: Figure height in pixels
        
    Returns:
        Ready-to-display Plotly Figure
    """
    viewer = Mesh3DViewer(mesh_data)
    
    if defects:
        positions = [d['position'] for d in defects]
        severities = [d.get('severity', 'medium') for d in defects]
        viewer.add_defect_markers(positions, severities=severities)
    
    fig = viewer.create_figure()
    fig.update_layout(height=height)
    
    return fig
