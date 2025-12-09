"""
Mesh loading and processing for 3D visualization.

Handles loading of OBJ/STL files using trimesh and preparing
data for Plotly rendering.
"""

import trimesh
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MeshData:
    """
    Container for processed mesh data.
    
    Attributes:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of face vertex indices
        face_normals: (M, 3) array of face normal vectors
        vertex_normals: (N, 3) array of vertex normal vectors
        bounds: Tuple of (min_corner, max_corner) arrays
        center: Center point of the mesh
        file_path: Original file path (if loaded from file)
        name: Display name for the mesh
    """
    vertices: np.ndarray
    faces: np.ndarray
    face_normals: np.ndarray
    vertex_normals: np.ndarray
    bounds: Tuple[np.ndarray, np.ndarray]
    center: np.ndarray
    file_path: Optional[str] = None
    name: str = "Untitled Part"


def load_mesh(file_path: str) -> MeshData:
    """
    Load an OBJ or STL file and extract mesh data.
    
    Args:
        file_path: Path to the mesh file (.obj or .stl)
        
    Returns:
        MeshData container with vertices, faces, and normals
        
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {file_path}")
    
    suffix = path.suffix.lower()
    if suffix not in ['.obj', '.stl']:
        raise ValueError(f"Unsupported mesh format: {suffix}. Use .obj or .stl")
    
    # Load with trimesh
    mesh = trimesh.load(file_path, force='mesh')
    
    # Handle scene vs single mesh
    if isinstance(mesh, trimesh.Scene):
        # Combine all meshes in scene
        meshes = list(mesh.geometry.values())
        if len(meshes) == 0:
            raise ValueError("No valid meshes found in file")
        mesh = trimesh.util.concatenate(meshes)
    
    # Ensure we have a proper mesh
    if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
        raise ValueError("Could not extract mesh data from file")
    
    # Center the mesh at origin
    mesh.vertices -= mesh.centroid
    
    # Compute bounds
    bounds_min = mesh.vertices.min(axis=0)
    bounds_max = mesh.vertices.max(axis=0)
    
    # Compute vertex normals if not available
    if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh.fix_normals()
    
    return MeshData(
        vertices=np.array(mesh.vertices, dtype=np.float32),
        faces=np.array(mesh.faces, dtype=np.int32),
        face_normals=np.array(mesh.face_normals, dtype=np.float32),
        vertex_normals=np.array(mesh.vertex_normals, dtype=np.float32),
        bounds=(bounds_min, bounds_max),
        center=np.array([0.0, 0.0, 0.0]),  # Centered at origin
        file_path=str(path.absolute()),
        name=path.stem
    )


def load_mesh_from_bytes(data: bytes, file_type: str, name: str = "Uploaded Part") -> MeshData:
    """
    Load mesh from bytes (for Streamlit file uploads).
    
    Args:
        data: Raw file bytes
        file_type: Either 'obj' or 'stl'
        name: Display name for the mesh
        
    Returns:
        MeshData container
    """
    import io
    
    file_type = file_type.lower().replace('.', '')
    if file_type not in ['obj', 'stl']:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    # Load from bytes
    mesh = trimesh.load(
        io.BytesIO(data),
        file_type=file_type,
        force='mesh'
    )
    
    # Handle scene vs single mesh
    if isinstance(mesh, trimesh.Scene):
        meshes = list(mesh.geometry.values())
        if len(meshes) == 0:
            raise ValueError("No valid meshes found in file")
        mesh = trimesh.util.concatenate(meshes)
    
    # Center the mesh
    mesh.vertices -= mesh.centroid
    
    # Compute bounds
    bounds_min = mesh.vertices.min(axis=0)
    bounds_max = mesh.vertices.max(axis=0)
    
    # Ensure normals
    if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh.fix_normals()
    
    return MeshData(
        vertices=np.array(mesh.vertices, dtype=np.float32),
        faces=np.array(mesh.faces, dtype=np.int32),
        face_normals=np.array(mesh.face_normals, dtype=np.float32),
        vertex_normals=np.array(mesh.vertex_normals, dtype=np.float32),
        bounds=(bounds_min, bounds_max),
        center=np.array([0.0, 0.0, 0.0]),
        file_path=None,
        name=name
    )


def get_mesh_scale(mesh_data: MeshData) -> float:
    """
    Calculate the maximum dimension of the mesh for scaling.
    
    Args:
        mesh_data: MeshData object
        
    Returns:
        Maximum dimension across X, Y, Z
    """
    dimensions = mesh_data.bounds[1] - mesh_data.bounds[0]
    return float(np.max(dimensions))


def get_mesh_dimensions(mesh_data: MeshData) -> Tuple[float, float, float]:
    """
    Get mesh dimensions (width, height, depth).
    
    Args:
        mesh_data: MeshData object
        
    Returns:
        Tuple of (x_size, y_size, z_size)
    """
    dims = mesh_data.bounds[1] - mesh_data.bounds[0]
    return tuple(dims)


def sample_surface_points(
    mesh_data: MeshData, 
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample random points on the mesh surface with their normals.
    
    Args:
        mesh_data: MeshData object
        n_points: Number of points to sample
        
    Returns:
        Tuple of (positions, normals) arrays, each (n_points, 3)
    """
    # Create trimesh from data for sampling
    mesh = trimesh.Trimesh(
        vertices=mesh_data.vertices,
        faces=mesh_data.faces
    )
    
    points, face_indices = mesh.sample(n_points, return_index=True)
    normals = mesh_data.face_normals[face_indices]
    
    return points, normals
