"""
Visualization module for interactive 3D mesh rendering.

Provides:
- MeshData: Container for processed mesh data
- load_mesh: Load OBJ/STL files using trimesh
- Mesh3DViewer: Interactive Plotly-based 3D viewer
- generate_test_meshes: Create sample industrial parts for testing
- generate_premium_meshes: Create high-quality industrial parts
"""

from .mesh_loader import MeshData, load_mesh, load_mesh_from_bytes, get_mesh_scale
from .plotly_viewer import Mesh3DViewer
from .test_meshes import generate_test_meshes, get_sample_defects
from .premium_meshes import generate_premium_meshes, get_premium_defects

__all__ = [
    'MeshData',
    'load_mesh',
    'load_mesh_from_bytes',
    'get_mesh_scale',
    'Mesh3DViewer',
    'generate_test_meshes',
    'get_sample_defects',
    'generate_premium_meshes',
    'get_premium_defects',
]
