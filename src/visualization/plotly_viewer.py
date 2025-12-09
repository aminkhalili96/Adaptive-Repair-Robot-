"""
Interactive 3D visualization with Plotly.

Creates browser-based 3D mesh viewer with:
- Mesh rendering with proper lighting
- Defect markers (points/spheres)
- Surface normal arrows
- Toolpath visualization
- Camera animation controls
"""

import plotly.graph_objects as go
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from src.visualization.mesh_loader import MeshData


# Claude-inspired dark theme colors
COLORS = {
    'background': '#1a1a1a',
    'mesh': '#3d3d3d',
    'mesh_edges': '#4a4a4a',
    'defect_high': '#e85d5d',      # Coral red for high severity
    'defect_medium': '#e8a55d',    # Orange for medium
    'defect_low': '#7dca9a',       # Green for low
    'normal_arrow': '#d97757',      # Warm coral (Claude accent)
    'toolpath': '#7db4ca',          # Soft blue
    'highlight': '#ffd700',         # Gold for highlights
    'text': '#ececec',
    'grid': '#333333',
}


@dataclass
class DefectMarker:
    """A defect to display on the mesh."""
    position: Tuple[float, float, float]
    label: str
    severity: str  # 'high', 'medium', 'low'
    normal: Optional[Tuple[float, float, float]] = None
    confidence: float = 1.0


class Mesh3DViewer:
    """
    Interactive 3D mesh viewer for Streamlit.
    
    Creates Plotly figures with mesh rendering, defect markers,
    surface normals, and toolpath visualization.
    """
    
    def __init__(self, mesh_data: MeshData):
        """
        Initialize viewer with mesh data.
        
        Args:
            mesh_data: MeshData object with vertices, faces, normals
        """
        self.mesh_data = mesh_data
        self.fig = None
        self._defect_markers: List[DefectMarker] = []
        self._toolpath_waypoints: List[Tuple[float, float, float]] = []
        self._highlight_position: Optional[Tuple[float, float, float]] = None
        self._highlight_radius: float = 0.05
        
        # Camera state for animation
        self._camera_eye = None
        self._camera_center = None
        
    def create_figure(self, show_edges: bool = False) -> go.Figure:
        """
        Create base Plotly figure with mesh.
        
        Args:
            show_edges: Whether to show mesh wireframe edges
            
        Returns:
            Plotly Figure object
        """
        vertices = self.mesh_data.vertices
        faces = self.mesh_data.faces
        
        # Create Mesh3d trace
        mesh_trace = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=COLORS['mesh'],
            opacity=0.95,
            flatshading=True,
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.3,
                roughness=0.5,
                fresnel=0.2
            ),
            lightposition=dict(x=100, y=100, z=200),
            hoverinfo='skip',
            name='Part'
        )
        
        traces = [mesh_trace]
        
        # Add edge lines if requested
        if show_edges:
            edge_trace = self._create_edge_trace()
            traces.append(edge_trace)
        
        # Create figure
        self.fig = go.Figure(data=traces)
        
        # Apply layout
        self._apply_dark_layout()
        
        # Add any defect markers
        if self._defect_markers:
            self._add_defect_traces()
        
        # Add toolpath if present
        if self._toolpath_waypoints:
            self._add_toolpath_trace()
        
        # Add highlight if present
        if self._highlight_position:
            self._add_highlight_trace()
        
        return self.fig
    
    def _create_edge_trace(self) -> go.Scatter3d:
        """Create wireframe edge trace."""
        vertices = self.mesh_data.vertices
        faces = self.mesh_data.faces
        
        # Extract edges
        edge_x, edge_y, edge_z = [], [], []
        for face in faces:
            for i in range(3):
                v0 = vertices[face[i]]
                v1 = vertices[face[(i + 1) % 3]]
                edge_x.extend([v0[0], v1[0], None])
                edge_y.extend([v0[1], v1[1], None])
                edge_z.extend([v0[2], v1[2], None])
        
        return go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color=COLORS['mesh_edges'], width=1),
            hoverinfo='skip',
            showlegend=False
        )
    
    def _apply_dark_layout(self):
        """Apply Claude-style dark theme layout."""
        # Calculate mesh bounds for axis ranges
        bounds = self.mesh_data.bounds
        padding = 0.1 * np.max(bounds[1] - bounds[0])
        
        x_range = [bounds[0][0] - padding, bounds[1][0] + padding]
        y_range = [bounds[0][1] - padding, bounds[1][1] + padding]
        z_range = [bounds[0][2] - padding, bounds[1][2] + padding]
        
        # Calculate camera position
        mesh_scale = np.max(bounds[1] - bounds[0])
        camera_distance = mesh_scale * 2.5
        
        self.fig.update_layout(
            scene=dict(
                xaxis=dict(
                    range=x_range,
                    showbackground=True,
                    backgroundcolor=COLORS['background'],
                    gridcolor=COLORS['grid'],
                    zerolinecolor=COLORS['grid'],
                    showspikes=False,
                    title='',
                    tickfont=dict(color=COLORS['text'], size=10),
                ),
                yaxis=dict(
                    range=y_range,
                    showbackground=True,
                    backgroundcolor=COLORS['background'],
                    gridcolor=COLORS['grid'],
                    zerolinecolor=COLORS['grid'],
                    showspikes=False,
                    title='',
                    tickfont=dict(color=COLORS['text'], size=10),
                ),
                zaxis=dict(
                    range=z_range,
                    showbackground=True,
                    backgroundcolor=COLORS['background'],
                    gridcolor=COLORS['grid'],
                    zerolinecolor=COLORS['grid'],
                    showspikes=False,
                    title='',
                    tickfont=dict(color=COLORS['text'], size=10),
                ),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
            ),
            paper_bgcolor=COLORS['background'],
            plot_bgcolor=COLORS['background'],
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(26, 26, 26, 0.8)',
                bordercolor=COLORS['grid'],
                borderwidth=1,
                font=dict(color=COLORS['text'], size=11),
            ),
            # Enable animation for camera transitions
            uirevision='constant',
        )
        
        # Store camera state
        self._camera_eye = dict(x=1.5, y=1.5, z=1.2)
        self._camera_center = dict(x=0, y=0, z=0)
    
    def add_defect_markers(
        self, 
        positions: List[Tuple[float, float, float]],
        labels: List[str],
        severities: List[str],
        normals: Optional[List[Tuple[float, float, float]]] = None,
        confidences: Optional[List[float]] = None
    ):
        """
        Add defect point markers to the view.
        
        Args:
            positions: List of (x, y, z) positions
            labels: List of defect type labels
            severities: List of severity levels ('high', 'medium', 'low')
            normals: Optional list of surface normals at each defect
            confidences: Optional list of detection confidences
        """
        for i, pos in enumerate(positions):
            marker = DefectMarker(
                position=pos,
                label=labels[i] if i < len(labels) else f"Defect {i+1}",
                severity=severities[i] if i < len(severities) else 'medium',
                normal=normals[i] if normals and i < len(normals) else None,
                confidence=confidences[i] if confidences and i < len(confidences) else 1.0
            )
            self._defect_markers.append(marker)
    
    def _add_defect_traces(self):
        """Add defect marker traces to figure."""
        if not self.fig:
            return
        
        # Group by severity for coloring
        for severity in ['high', 'medium', 'low']:
            markers = [m for m in self._defect_markers if m.severity == severity]
            if not markers:
                continue
            
            color = COLORS[f'defect_{severity}']
            
            # Add marker points
            self.fig.add_trace(go.Scatter3d(
                x=[m.position[0] for m in markers],
                y=[m.position[1] for m in markers],
                z=[m.position[2] for m in markers],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=color,
                    symbol='diamond',
                    line=dict(color='white', width=1),
                ),
                text=[m.label for m in markers],
                textposition='top center',
                textfont=dict(color=COLORS['text'], size=11),
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'Position: (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>'
                    f'Severity: {severity.capitalize()}<br>'
                    '<extra></extra>'
                ),
                name=f'{severity.capitalize()} Severity',
                showlegend=True
            ))
        
        # Add normal arrows for defects with normals
        self._add_normal_arrows()
    
    def _add_normal_arrows(self, scale: float = 0.04):
        """Add surface normal arrows at defect locations."""
        if not self.fig:
            return
        
        markers_with_normals = [m for m in self._defect_markers if m.normal]
        if not markers_with_normals:
            return
        
        # Create cone traces for arrows
        for marker in markers_with_normals:
            pos = np.array(marker.position)
            normal = np.array(marker.normal)
            
            # Arrow tip
            tip = pos + normal * scale
            
            self.fig.add_trace(go.Cone(
                x=[pos[0]],
                y=[pos[1]],
                z=[pos[2]],
                u=[normal[0] * scale],
                v=[normal[1] * scale],
                w=[normal[2] * scale],
                colorscale=[[0, COLORS['normal_arrow']], [1, COLORS['normal_arrow']]],
                sizemode='absolute',
                sizeref=scale * 0.3,
                showscale=False,
                hoverinfo='skip',
                showlegend=False
            ))
    
    def add_normal_arrows_at_points(
        self,
        positions: List[Tuple[float, float, float]],
        normals: List[Tuple[float, float, float]],
        scale: float = 0.03
    ):
        """
        Draw surface normal arrows at arbitrary positions.
        
        Args:
            positions: List of (x, y, z) base positions
            normals: List of (nx, ny, nz) normal vectors
            scale: Arrow length scale
        """
        if not self.fig or not positions:
            return
        
        for pos, normal in zip(positions, normals):
            pos = np.array(pos)
            normal = np.array(normal)
            
            self.fig.add_trace(go.Cone(
                x=[pos[0]],
                y=[pos[1]],
                z=[pos[2]],
                u=[normal[0] * scale],
                v=[normal[1] * scale],
                w=[normal[2] * scale],
                colorscale=[[0, COLORS['normal_arrow']], [1, COLORS['normal_arrow']]],
                sizemode='absolute',
                sizeref=scale * 0.3,
                showscale=False,
                hoverinfo='skip',
                showlegend=False
            ))
    
    def add_toolpath(self, waypoints: List[Tuple[float, float, float]]):
        """
        Draw toolpath as lines on the mesh.
        
        Args:
            waypoints: List of (x, y, z) waypoint positions
        """
        self._toolpath_waypoints = waypoints
    
    def _add_toolpath_trace(self):
        """Add toolpath line trace to figure."""
        if not self.fig or not self._toolpath_waypoints:
            return
        
        waypoints = self._toolpath_waypoints
        
        # Create line trace
        self.fig.add_trace(go.Scatter3d(
            x=[w[0] for w in waypoints],
            y=[w[1] for w in waypoints],
            z=[w[2] for w in waypoints],
            mode='lines+markers',
            line=dict(
                color=COLORS['toolpath'],
                width=4,
            ),
            marker=dict(
                size=3,
                color=COLORS['toolpath'],
            ),
            name='Toolpath',
            hovertemplate=(
                'Waypoint<br>'
                'Position: (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>'
                '<extra></extra>'
            ),
        ))
        
        # Add start/end markers
        if len(waypoints) >= 2:
            self.fig.add_trace(go.Scatter3d(
                x=[waypoints[0][0]],
                y=[waypoints[0][1]],
                z=[waypoints[0][2]],
                mode='markers',
                marker=dict(size=10, color='#7dca9a', symbol='circle'),
                name='Start',
                showlegend=False
            ))
            self.fig.add_trace(go.Scatter3d(
                x=[waypoints[-1][0]],
                y=[waypoints[-1][1]],
                z=[waypoints[-1][2]],
                mode='markers',
                marker=dict(size=10, color=COLORS['defect_high'], symbol='square'),
                name='End',
                showlegend=False
            ))
    
    def set_camera_view(
        self, 
        target_position: Tuple[float, float, float],
        distance: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate camera parameters to look at a specific position.
        
        This returns the camera dict for Plotly layout update,
        which can be used with animation frames.
        
        Args:
            target_position: Position to focus on
            distance: Distance from target (auto-calculated if None)
            
        Returns:
            Camera dict for plotly layout
        """
        target = np.array(target_position)
        
        # Calculate distance based on mesh scale if not provided
        if distance is None:
            mesh_scale = np.max(self.mesh_data.bounds[1] - self.mesh_data.bounds[0])
            distance = mesh_scale * 1.5
        
        # Calculate camera position (45-degree view from above)
        eye = target + np.array([distance * 0.7, distance * 0.7, distance * 0.5])
        
        camera = dict(
            eye=dict(x=eye[0], y=eye[1], z=eye[2]),
            center=dict(x=target[0], y=target[1], z=target[2]),
            up=dict(x=0, y=0, z=1)
        )
        
        self._camera_eye = camera['eye']
        self._camera_center = camera['center']
        
        return camera
    
    def highlight_region(
        self, 
        center: Tuple[float, float, float], 
        radius: float = 0.05
    ):
        """
        Highlight a spherical region around a point.
        
        Args:
            center: Center position to highlight
            radius: Radius of highlight sphere
        """
        self._highlight_position = center
        self._highlight_radius = radius
    
    def _add_highlight_trace(self):
        """Add highlight sphere trace to figure."""
        if not self.fig or not self._highlight_position:
            return
        
        center = self._highlight_position
        r = self._highlight_radius
        
        # Create sphere points
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x = center[0] + r * np.outer(np.cos(u), np.sin(v))
        y = center[1] + r * np.outer(np.sin(u), np.sin(v))
        z = center[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, COLORS['highlight']], [1, COLORS['highlight']]],
            opacity=0.3,
            showscale=False,
            hoverinfo='skip',
            showlegend=False
        ))
    
    def get_animation_frames(
        self,
        target_position: Tuple[float, float, float],
        n_frames: int = 30
    ) -> List[go.Frame]:
        """
        Generate animation frames for camera transition.
        
        Args:
            target_position: Final camera target
            n_frames: Number of animation frames
            
        Returns:
            List of Plotly Frame objects
        """
        if not self._camera_eye or not self._camera_center:
            return []
        
        start_eye = np.array([
            self._camera_eye['x'],
            self._camera_eye['y'],
            self._camera_eye['z']
        ])
        start_center = np.array([
            self._camera_center['x'],
            self._camera_center['y'],
            self._camera_center['z']
        ])
        
        end_camera = self.set_camera_view(target_position)
        end_eye = np.array([
            end_camera['eye']['x'],
            end_camera['eye']['y'],
            end_camera['eye']['z']
        ])
        end_center = np.array([
            end_camera['center']['x'],
            end_camera['center']['y'],
            end_camera['center']['z']
        ])
        
        frames = []
        for i in range(n_frames):
            t = i / (n_frames - 1)
            # Smooth easing
            t = t * t * (3 - 2 * t)
            
            eye = start_eye + t * (end_eye - start_eye)
            center = start_center + t * (end_center - start_center)
            
            frame = go.Frame(
                layout=dict(
                    scene_camera=dict(
                        eye=dict(x=eye[0], y=eye[1], z=eye[2]),
                        center=dict(x=center[0], y=center[1], z=center[2]),
                        up=dict(x=0, y=0, z=1)
                    )
                ),
                name=f'frame_{i}'
            )
            frames.append(frame)
        
        return frames
    
    def update_camera(self, target_position: Tuple[float, float, float]):
        """
        Update figure camera to look at target (instant, no animation).
        
        Args:
            target_position: Position to focus on
        """
        if not self.fig:
            return
        
        camera = self.set_camera_view(target_position)
        self.fig.update_layout(scene_camera=camera)
    
    def clear_overlays(self):
        """Clear all defect markers, toolpaths, and highlights."""
        self._defect_markers = []
        self._toolpath_waypoints = []
        self._highlight_position = None
