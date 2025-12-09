"""
Generate premium test meshes for demonstration.

Creates industrial-quality 3D parts for testing:
- Turbine blade with curved surfaces
- Pipe connector with flanges
- Mounting bracket with complex geometry
- Gear with teeth and hub
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Tuple, List


def _create_turbine_blade() -> trimesh.Trimesh:
    """
    Create a turbine blade with curved aerodynamic surfaces.
    
    Returns:
        Trimesh object representing a turbine blade
    """
    # Blade profile - airfoil cross-section
    n_profile = 30
    n_span = 20
    
    # NACA-like airfoil shape (simplified)
    t = np.linspace(0, 2 * np.pi, n_profile)
    
    # Chord varies along span (tapered blade)
    span_positions = np.linspace(0, 1, n_span)
    
    vertices = []
    faces = []
    
    for i, span in enumerate(span_positions):
        # Taper and twist along span
        chord = 0.12 * (1 - 0.4 * span)  # Chord decreases toward tip
        twist = span * 0.3  # Twist angle increases toward tip
        
        # Airfoil shape (thickness varies with span)
        thickness = 0.03 * (1 - 0.5 * span)
        
        for j, angle in enumerate(t[:-1]):  # Skip last (duplicate of first)
            # Airfoil cross-section
            x_local = chord * (0.5 - 0.5 * np.cos(angle))
            y_local = thickness * np.sin(angle) * (1 - (2 * x_local / chord - 1) ** 2)
            
            # Apply twist
            x_twisted = x_local * np.cos(twist) - y_local * np.sin(twist)
            y_twisted = x_local * np.sin(twist) + y_local * np.cos(twist)
            
            # Position along span (z direction, with slight sweep)
            z = span * 0.25 + x_local * 0.05
            x = x_twisted - chord * 0.25
            y = y_twisted
            
            vertices.append([x, y, z])
    
    # Create faces connecting profile rings
    n_ring = n_profile - 1
    for i in range(n_span - 1):
        for j in range(n_ring):
            v0 = i * n_ring + j
            v1 = i * n_ring + (j + 1) % n_ring
            v2 = (i + 1) * n_ring + j
            v3 = (i + 1) * n_ring + (j + 1) % n_ring
            
            faces.append([v0, v1, v3])
            faces.append([v0, v3, v2])
    
    # Cap the root (base of blade)
    root_center = len(vertices)
    vertices.append([0, 0, 0])
    for j in range(n_ring):
        v0 = j
        v1 = (j + 1) % n_ring
        faces.append([root_center, v1, v0])
    
    # Cap the tip
    tip_center = len(vertices)
    tip_z = 0.25 + 0.12 * 0.05
    vertices.append([0, 0, tip_z])
    tip_start = (n_span - 1) * n_ring
    for j in range(n_ring):
        v0 = tip_start + j
        v1 = tip_start + (j + 1) % n_ring
        faces.append([tip_center, v0, v1])
    
    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
    mesh.fix_normals()
    
    return mesh


def _create_pipe_connector() -> trimesh.Trimesh:
    """
    Create a pipe connector with flanges and mounting holes.
    
    Returns:
        Trimesh object representing a pipe connector
    """
    # Main pipe body - cylinder
    pipe = trimesh.creation.cylinder(radius=0.04, height=0.15, sections=32)
    pipe.apply_translation([0, 0, 0.075])
    
    # Input flange (larger disk at bottom)
    flange1 = trimesh.creation.cylinder(radius=0.07, height=0.015, sections=32)
    flange1.apply_translation([0, 0, 0.0075])
    
    # Output flange (at top)
    flange2 = trimesh.creation.cylinder(radius=0.065, height=0.012, sections=32)
    flange2.apply_translation([0, 0, 0.144])
    
    # Mounting holes in flanges (decorative cylinders for visual)
    holes = []
    for i in range(6):
        angle = i * 2 * np.pi / 6
        x = 0.055 * np.cos(angle)
        y = 0.055 * np.sin(angle)
        hole = trimesh.creation.cylinder(radius=0.006, height=0.018, sections=16)
        hole.apply_translation([x, y, 0.009])
        holes.append(hole)
    
    # Side port (T-junction style)
    side_port = trimesh.creation.cylinder(radius=0.025, height=0.05, sections=24)
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    side_port.apply_transform(rot)
    side_port.apply_translation([0, 0.065, 0.08])
    
    # Side port flange
    side_flange = trimesh.creation.cylinder(radius=0.04, height=0.008, sections=24)
    side_flange.apply_transform(rot)
    side_flange.apply_translation([0, 0.09, 0.08])
    
    # Combine all parts
    result = trimesh.util.concatenate([pipe, flange1, flange2, side_port, side_flange] + holes)
    result.fix_normals()
    
    return result


def _create_mounting_bracket() -> trimesh.Trimesh:
    """
    Create an L-shaped mounting bracket with reinforcement.
    
    Returns:
        Trimesh object representing a mounting bracket
    """
    # Base plate
    base = trimesh.creation.box([0.15, 0.08, 0.01])
    base.apply_translation([0.075, 0.04, 0.005])
    
    # Vertical plate
    vertical = trimesh.creation.box([0.01, 0.08, 0.12])
    vertical.apply_translation([0.005, 0.04, 0.06 + 0.01])
    
    # Diagonal reinforcement (triangular)
    vertices = np.array([
        [0.01, 0.01, 0.01],   # Bottom front
        [0.01, 0.07, 0.01],   # Bottom back
        [0.06, 0.01, 0.01],   # Bottom right front
        [0.06, 0.07, 0.01],   # Bottom right back
        [0.01, 0.01, 0.08],   # Top front
        [0.01, 0.07, 0.08],   # Top back
    ])
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2],
        [0, 2, 4],
        [2, 4, 4],  # Fixed below
        [1, 5, 3],
        [3, 5, 5],  # Fixed below
        [0, 4, 1],
        [1, 4, 5],
        [2, 3, 4],
        [3, 5, 4],
    ])
    # Create proper triangular gusset
    gusset = trimesh.creation.box([0.05, 0.06, 0.002])
    rot = trimesh.transformations.rotation_matrix(np.pi / 4, [0, 1, 0])
    gusset.apply_transform(rot)
    gusset.apply_translation([0.035, 0.04, 0.045])
    
    # Mounting holes (cylindrical)
    hole1 = trimesh.creation.cylinder(radius=0.008, height=0.02, sections=16)
    hole1.apply_translation([0.03, 0.04, 0.005])
    
    hole2 = trimesh.creation.cylinder(radius=0.008, height=0.02, sections=16)
    hole2.apply_translation([0.10, 0.04, 0.005])
    
    hole3 = trimesh.creation.cylinder(radius=0.008, height=0.02, sections=16)
    rot_v = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    hole3.apply_transform(rot_v)
    hole3.apply_translation([0.005, 0.04, 0.10])
    
    # Rounded edge at corner (aesthetic)
    corner = trimesh.creation.cylinder(radius=0.01, height=0.08, sections=16)
    corner.apply_translation([0.01, 0.04, 0.01])
    
    result = trimesh.util.concatenate([base, vertical, gusset, hole1, hole2, hole3, corner])
    result.fix_normals()
    
    return result


def _create_gear() -> trimesh.Trimesh:
    """
    Create a spur gear with teeth and central hub.
    
    Returns:
        Trimesh object representing a gear
    """
    n_teeth = 16
    outer_radius = 0.06
    inner_radius = 0.045
    hub_radius = 0.025
    thickness = 0.015
    hub_thickness = 0.025
    
    # Create main gear body as cylinder with teeth approximated
    # by combining multiple shapes
    
    # Base disk
    base_disk = trimesh.creation.cylinder(
        radius=inner_radius, 
        height=thickness, 
        sections=64
    )
    base_disk.apply_translation([0, 0, (hub_thickness - thickness) / 2 + thickness / 2])
    
    # Create teeth as small boxes around circumference
    teeth = []
    for i in range(n_teeth):
        angle = i * 2 * np.pi / n_teeth
        
        # Tooth dimensions
        tooth_width = 0.012
        tooth_height = 0.015  # Radial extension
        
        # Create tooth as box
        tooth = trimesh.creation.box([tooth_height, tooth_width, thickness])
        
        # Position tooth
        x = (inner_radius + tooth_height / 2) * np.cos(angle)
        y = (inner_radius + tooth_height / 2) * np.sin(angle)
        z = (hub_thickness - thickness) / 2 + thickness / 2
        
        # Rotate to align with radius
        rot = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        tooth.apply_transform(rot)
        tooth.apply_translation([x, y, z])
        
        teeth.append(tooth)
    
    # Central hub (thicker cylinder in center)
    hub = trimesh.creation.cylinder(radius=hub_radius, height=hub_thickness, sections=32)
    hub.apply_translation([0, 0, hub_thickness / 2])
    
    # Central bore
    bore = trimesh.creation.cylinder(radius=0.008, height=hub_thickness + 0.002, sections=24)
    bore.apply_translation([0, 0, hub_thickness / 2])
    
    # Keyway (small rectangular notch)
    keyway = trimesh.creation.box([0.004, 0.016, hub_thickness])
    keyway.apply_translation([0.008, 0, hub_thickness / 2])
    
    # Lightening holes (for weight reduction, visual detail)
    holes = []
    n_holes = 4
    hole_radius = 0.008
    hole_center_radius = 0.035
    for i in range(n_holes):
        angle = i * 2 * np.pi / n_holes + np.pi / n_holes
        x = hole_center_radius * np.cos(angle)
        y = hole_center_radius * np.sin(angle)
        hole = trimesh.creation.cylinder(radius=hole_radius, height=thickness + 0.002, sections=24)
        hole.apply_translation([x, y, (hub_thickness - thickness) / 2 + thickness / 2])
        holes.append(hole)
    
    # Combine all parts
    result = trimesh.util.concatenate([base_disk, hub, keyway, bore] + teeth + holes)
    result.fix_normals()
    
    return result


def generate_test_meshes(output_dir: str) -> List[str]:
    """
    Generate all test meshes and save to directory.
    
    Args:
        output_dir: Directory to save mesh files
        
    Returns:
        List of created file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    meshes = {
        'turbine_blade': _create_turbine_blade(),
        'pipe_connector': _create_pipe_connector(),
        'mounting_bracket': _create_mounting_bracket(),
        'gear': _create_gear(),
    }
    
    created_files = []
    
    for name, mesh in meshes.items():
        # Save as both STL and OBJ
        stl_path = output_path / f"{name}.stl"
        obj_path = output_path / f"{name}.obj"
        
        mesh.export(str(stl_path))
        mesh.export(str(obj_path))
        
        created_files.append(str(stl_path))
        created_files.append(str(obj_path))
        
        print(f"Created: {name}")
        print(f"  - Vertices: {len(mesh.vertices)}")
        print(f"  - Faces: {len(mesh.faces)}")
        print(f"  - Bounds: {mesh.bounds}")
    
    return created_files


def get_sample_defects(mesh_name: str) -> List[dict]:
    """
    Get predefined sample defects for demo meshes.
    
    Args:
        mesh_name: Name of the mesh (e.g., 'turbine_blade')
        
    Returns:
        List of defect dicts with position, type, severity
    """
    defects_map = {
        'turbine_blade': [
            {'position': (0.02, 0.01, 0.15), 'type': 'crack', 'severity': 'high', 'confidence': 0.92},
            {'position': (-0.03, 0.005, 0.08), 'type': 'erosion', 'severity': 'medium', 'confidence': 0.85},
            {'position': (0.01, -0.01, 0.22), 'type': 'pitting', 'severity': 'low', 'confidence': 0.78},
        ],
        'pipe_connector': [
            {'position': (0.04, 0.0, 0.12), 'type': 'corrosion', 'severity': 'high', 'confidence': 0.95},
            {'position': (0.0, 0.07, 0.08), 'type': 'crack', 'severity': 'medium', 'confidence': 0.88},
            {'position': (-0.02, 0.03, 0.01), 'type': 'rust', 'severity': 'low', 'confidence': 0.72},
        ],
        'mounting_bracket': [
            {'position': (0.01, 0.04, 0.08), 'type': 'fatigue_crack', 'severity': 'high', 'confidence': 0.91},
            {'position': (0.1, 0.04, 0.005), 'type': 'wear', 'severity': 'medium', 'confidence': 0.82},
        ],
        'gear': [
            {'position': (0.05, 0.0, 0.012), 'type': 'tooth_wear', 'severity': 'high', 'confidence': 0.93},
            {'position': (-0.04, 0.03, 0.012), 'type': 'pitting', 'severity': 'medium', 'confidence': 0.79},
            {'position': (0.0, -0.05, 0.012), 'type': 'crack', 'severity': 'low', 'confidence': 0.71},
        ],
    }
    
    return defects_map.get(mesh_name, [])


if __name__ == '__main__':
    # Generate test meshes when run directly
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else './test_meshes'
    generate_test_meshes(output_dir)
