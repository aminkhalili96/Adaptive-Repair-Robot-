"""
Premium industrial test mesh generator.

Creates high-quality, visually impressive 3D parts:
- High polygon count for smooth surfaces
- Realistic industrial proportions
- Complex geometry with bevels and details
- Proper mesh topology for rendering
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import List, Tuple
from scipy.spatial.transform import Rotation


def create_industrial_turbine_blade() -> trimesh.Trimesh:
    """
    Create a high-quality turbine blade with realistic airfoil profile.
    
    Features:
    - NACA-style airfoil cross-section
    - Proper twist and taper
    - Fillet at root
    - High polygon count for smooth rendering
    """
    # High resolution for smooth surfaces
    n_span = 60  # Spanwise sections
    n_profile = 80  # Profile points per section
    
    chord_root = 0.12
    chord_tip = 0.04
    span_length = 0.35
    max_thickness = 0.025
    
    vertices = []
    
    for i in range(n_span):
        # Spanwise position (0 = root, 1 = tip)
        t = i / (n_span - 1)
        
        # Chord varies along span (tapered)
        chord = chord_root * (1 - t) + chord_tip * t
        
        # Twist increases toward tip
        twist = t * 25 * np.pi / 180  # 0 to 25 degrees
        
        # Thickness distribution (thicker at root)
        thickness = max_thickness * (1 - 0.5 * t)
        
        # Span position with slight sweep
        z = t * span_length
        sweep_offset = t * t * 0.02  # Quadratic sweep
        
        # Generate NACA 4-digit style airfoil profile
        for j in range(n_profile):
            # Parameter around airfoil (0 to 2*pi)
            theta = 2 * np.pi * j / n_profile
            
            # Airfoil coordinates (NACA-like)
            # Upper surface
            if theta <= np.pi:
                x_local = 0.5 * chord * (1 - np.cos(theta))
                # Thickness distribution: max at 30% chord
                x_norm = x_local / chord
                y_local = thickness * (
                    0.2969 * np.sqrt(x_norm) 
                    - 0.1260 * x_norm 
                    - 0.3516 * x_norm**2 
                    + 0.2843 * x_norm**3 
                    - 0.1015 * x_norm**4
                ) * 5
            else:
                # Lower surface (mirror)
                x_local = 0.5 * chord * (1 - np.cos(2*np.pi - theta))
                x_norm = x_local / chord
                y_local = -thickness * (
                    0.2969 * np.sqrt(x_norm + 1e-6) 
                    - 0.1260 * x_norm 
                    - 0.3516 * x_norm**2 
                    + 0.2843 * x_norm**3 
                    - 0.1015 * x_norm**4
                ) * 5
            
            # Apply twist
            x_twisted = (x_local - chord/2) * np.cos(twist) - y_local * np.sin(twist)
            y_twisted = (x_local - chord/2) * np.sin(twist) + y_local * np.cos(twist)
            
            # Final position
            x = x_twisted + sweep_offset
            y = y_twisted
            
            vertices.append([x, y, z])
    
    # Create faces
    faces = []
    for i in range(n_span - 1):
        for j in range(n_profile):
            v0 = i * n_profile + j
            v1 = i * n_profile + (j + 1) % n_profile
            v2 = (i + 1) * n_profile + j
            v3 = (i + 1) * n_profile + (j + 1) % n_profile
            
            faces.append([v0, v1, v3])
            faces.append([v0, v3, v2])
    
    # Cap root
    root_center_idx = len(vertices)
    vertices.append([0, 0, 0])
    for j in range(n_profile):
        v0 = j
        v1 = (j + 1) % n_profile
        faces.append([root_center_idx, v1, v0])
    
    # Cap tip
    tip_center_idx = len(vertices)
    vertices.append([0.02, 0, span_length])
    tip_start = (n_span - 1) * n_profile
    for j in range(n_profile):
        v0 = tip_start + j
        v1 = tip_start + (j + 1) % n_profile
        faces.append([tip_center_idx, v0, v1])
    
    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
    mesh.fix_normals()
    
    return mesh


def create_industrial_pipe_assembly() -> trimesh.Trimesh:
    """
    Create a realistic pipe assembly with flanges, bolts, and weld beads.
    Uses only additive geometry (no boolean operations).
    """
    pipe_od = 0.06
    pipe_length = 0.15
    flange_od = 0.10
    flange_thickness = 0.012
    bolt_circle = 0.08
    n_bolts = 8
    
    sections = 64
    
    parts = []
    
    # Main pipe body (solid cylinder - simpler)
    pipe_body = trimesh.creation.cylinder(radius=pipe_od/2, height=pipe_length, sections=sections)
    pipe_body.apply_translation([0, 0, pipe_length/2])
    parts.append(pipe_body)
    
    # Input flange (bottom)
    flange1 = trimesh.creation.cylinder(radius=flange_od/2, height=flange_thickness, sections=sections)
    flange1.apply_translation([0, 0, flange_thickness/2])
    parts.append(flange1)
    
    # Bolt studs around bottom flange
    for i in range(n_bolts):
        angle = 2 * np.pi * i / n_bolts
        x = bolt_circle/2 * np.cos(angle)
        y = bolt_circle/2 * np.sin(angle)
        stud = trimesh.creation.cylinder(radius=0.004, height=0.02, sections=12)
        stud.apply_translation([x, y, flange_thickness + 0.01])
        parts.append(stud)
        # Nut
        nut = trimesh.creation.cylinder(radius=0.007, height=0.006, sections=6)
        nut.apply_translation([x, y, flange_thickness + 0.017])
        parts.append(nut)
    
    # Output flange (top)
    flange2 = trimesh.creation.cylinder(radius=flange_od/2, height=flange_thickness, sections=sections)
    flange2.apply_translation([0, 0, pipe_length + flange_thickness/2])
    parts.append(flange2)
    
    # Bolt studs on top flange
    for i in range(n_bolts):
        angle = 2 * np.pi * i / n_bolts
        x = bolt_circle/2 * np.cos(angle)
        y = bolt_circle/2 * np.sin(angle)
        stud = trimesh.creation.cylinder(radius=0.004, height=0.02, sections=12)
        stud.apply_translation([x, y, pipe_length + flange_thickness + 0.01])
        parts.append(stud)
        nut = trimesh.creation.cylinder(radius=0.007, height=0.006, sections=6)
        nut.apply_translation([x, y, pipe_length + flange_thickness + 0.017])
        parts.append(nut)
    
    # Weld beads (tori at flange-pipe joints)
    weld1 = trimesh.creation.torus(major_radius=pipe_od/2 + 0.002, minor_radius=0.004)
    weld1.apply_translation([0, 0, flange_thickness])
    parts.append(weld1)
    
    weld2 = trimesh.creation.torus(major_radius=pipe_od/2 + 0.002, minor_radius=0.004)
    weld2.apply_translation([0, 0, pipe_length])
    parts.append(weld2)
    
    # Side branch (T-junction)
    branch_length = 0.07
    branch = trimesh.creation.cylinder(radius=0.025, height=branch_length, sections=sections)
    rot = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    branch.apply_transform(rot)
    branch.apply_translation([0, branch_length/2 + pipe_od/2, pipe_length * 0.6])
    parts.append(branch)
    
    # Branch flange
    branch_flange = trimesh.creation.cylinder(radius=0.04, height=0.008, sections=sections)
    branch_flange.apply_transform(rot)
    branch_flange.apply_translation([0, branch_length + pipe_od/2 + 0.004, pipe_length * 0.6])
    parts.append(branch_flange)
    
    # Branch weld bead
    branch_weld = trimesh.creation.torus(major_radius=0.027, minor_radius=0.003)
    branch_weld.apply_transform(rot)
    branch_weld.apply_translation([0, pipe_od/2 + 0.002, pipe_length * 0.6])
    parts.append(branch_weld)
    
    # Reinforcement pad at branch junction
    pad = trimesh.creation.cylinder(radius=0.035, height=0.003, sections=32)
    pad.apply_transform(rot)
    pad.apply_translation([0, pipe_od/2 + 0.0015, pipe_length * 0.6])
    parts.append(pad)
    
    result = trimesh.util.concatenate(parts)
    result.fix_normals()
    
    return result


def create_precision_gear() -> trimesh.Trimesh:
    """
    Create a high-quality involute spur gear.
    Uses proper involute tooth profile approximation.
    """
    n_teeth = 24
    m = 0.003  # Module in meters
    pressure_angle = 20 * np.pi / 180
    
    pitch_radius = n_teeth * m / 2
    base_radius = pitch_radius * np.cos(pressure_angle)
    addendum = m
    dedendum = 1.25 * m
    outer_radius = pitch_radius + addendum
    root_radius = pitch_radius - dedendum
    
    face_width = 0.02
    hub_radius = 0.018
    hub_length = 0.03
    
    # High resolution for smooth gear profile
    n_points_per_tooth = 16
    n_points = n_teeth * n_points_per_tooth
    
    # Build gear profile vertices
    profile_points = []
    
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        tooth_phase = (angle * n_teeth / (2 * np.pi)) % 1.0
        
        # Tooth profile with involute approximation
        if tooth_phase < 0.15:
            # Root fillet
            t = tooth_phase / 0.15
            r = root_radius + (pitch_radius - root_radius - 0.001) * (1 - np.cos(t * np.pi / 2))
        elif tooth_phase < 0.35:
            # Involute flank (rising)
            t = (tooth_phase - 0.15) / 0.20
            r = pitch_radius + (outer_radius - pitch_radius) * np.sin(t * np.pi / 2)
        elif tooth_phase < 0.50:
            # Tooth tip
            r = outer_radius
        elif tooth_phase < 0.70:
            # Involute flank (falling)
            t = (tooth_phase - 0.50) / 0.20
            r = outer_radius - (outer_radius - pitch_radius) * np.sin(t * np.pi / 2)
        elif tooth_phase < 0.85:
            # Root fillet
            t = (tooth_phase - 0.70) / 0.15
            r = pitch_radius - (pitch_radius - root_radius - 0.001) * np.sin(t * np.pi / 2)
        else:
            # Root
            r = root_radius
        
        profile_points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Create extruded gear profile
    vertices = []
    faces = []
    
    # Bottom face vertices
    for pt in profile_points:
        vertices.append([pt[0], pt[1], 0])
    
    # Top face vertices
    for pt in profile_points:
        vertices.append([pt[0], pt[1], face_width])
    
    # Side faces
    for i in range(n_points):
        v0 = i
        v1 = (i + 1) % n_points
        v2 = n_points + i
        v3 = n_points + (i + 1) % n_points
        
        faces.append([v0, v2, v1])
        faces.append([v1, v2, v3])
    
    # Bottom cap
    center_bottom = len(vertices)
    vertices.append([0, 0, 0])
    for i in range(n_points):
        v0 = i
        v1 = (i + 1) % n_points
        faces.append([center_bottom, v1, v0])
    
    # Top cap
    center_top = len(vertices)
    vertices.append([0, 0, face_width])
    for i in range(n_points):
        v0 = n_points + i
        v1 = n_points + (i + 1) % n_points
        faces.append([center_top, v0, v1])
    
    gear_body = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
    
    parts = [gear_body]
    
    # Central hub
    hub = trimesh.creation.cylinder(radius=hub_radius, height=hub_length, sections=64)
    hub.apply_translation([0, 0, face_width/2])
    parts.append(hub)
    
    # Keyway detail
    keyway = trimesh.creation.box([0.005, 0.015, hub_length])
    keyway.apply_translation([hub_radius - 0.0025, 0, face_width/2])
    parts.append(keyway)
    
    # Chamfer on hub top
    chamfer = trimesh.creation.cylinder(radius=hub_radius + 0.002, height=0.002, sections=64)
    chamfer.apply_translation([0, 0, face_width/2 + hub_length/2 - 0.001])
    parts.append(chamfer)
    
    result = trimesh.util.concatenate(parts)
    result.fix_normals()
    
    return result


def create_aerospace_bracket() -> trimesh.Trimesh:
    """
    Create a premium aerospace-style mounting bracket.
    Uses additive geometry for complex features.
    """
    base_length = 0.12
    base_width = 0.07
    base_thickness = 0.005
    
    wall_height = 0.10
    wall_thickness = 0.005
    
    parts = []
    
    # Main base plate
    base = trimesh.creation.box([base_length, base_width, base_thickness])
    base.apply_translation([base_length/2, base_width/2, base_thickness/2])
    parts.append(base)
    
    # Vertical wall
    wall = trimesh.creation.box([wall_thickness, base_width, wall_height])
    wall.apply_translation([wall_thickness/2, base_width/2, base_thickness + wall_height/2])
    parts.append(wall)
    
    # Gusset ribs (triangular reinforcement)
    n_ribs = 3
    for i in range(n_ribs):
        y_pos = base_width * (i + 1) / (n_ribs + 1)
        
        # Create rib as angled box
        rib = trimesh.creation.box([0.045, 0.004, 0.005])
        
        # Rotate 45 degrees
        rot = trimesh.transformations.rotation_matrix(-np.pi/4, [0, 1, 0])
        rib.apply_transform(rot)
        rib.apply_translation([0.025, y_pos, base_thickness + 0.025])
        parts.append(rib)
    
    # Mounting bosses with countersink effect
    boss_positions = [
        (0.025, 0.015),
        (0.025, base_width - 0.015),
        (base_length - 0.02, 0.015),
        (base_length - 0.02, base_width - 0.015),
    ]
    
    for x, y in boss_positions:
        # Raised boss
        boss = trimesh.creation.cylinder(radius=0.008, height=0.003, sections=64)
        boss.apply_translation([x, y, base_thickness + 0.0015])
        parts.append(boss)
    
    # Wall mounting bosses
    for z in [base_thickness + 0.025, base_thickness + 0.06, base_thickness + 0.085]:
        boss = trimesh.creation.cylinder(radius=0.006, height=0.003, sections=64)
        rot = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
        boss.apply_transform(rot)
        boss.apply_translation([wall_thickness + 0.0015, base_width/2, z])
        parts.append(boss)
    
    # Top flange for rigidity
    top_flange = trimesh.creation.box([0.015, base_width - 0.01, wall_thickness])
    top_flange.apply_translation([wall_thickness/2 + 0.0075, base_width/2, base_thickness + wall_height - wall_thickness/2])
    parts.append(top_flange)
    
    # Corner fillet (quarter cylinder)
    fillet = trimesh.creation.cylinder(radius=0.012, height=base_width - 0.02, sections=64)
    rot = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    fillet.apply_transform(rot)
    fillet.apply_translation([wall_thickness + 0.006, base_width/2, base_thickness + 0.008])
    parts.append(fillet)
    
    # Stiffener flange along base
    stiffener = trimesh.creation.box([base_length - wall_thickness - 0.02, 0.004, 0.008])
    stiffener.apply_translation([wall_thickness + (base_length - wall_thickness - 0.02)/2 + 0.01, base_width/2, base_thickness + 0.004])
    parts.append(stiffener)
    
    result = trimesh.util.concatenate(parts)
    result.fix_normals()
    
    return result


def create_robotic_gripper() -> trimesh.Trimesh:
    """
    Create a high-fidelity robotic parallel gripper assembly.
    
    Features:
    - Billet-style base with rounded edges
    - Dual precision rails and sliding jaws
    - Pneumatic actuator with clevis
    - Fastener bosses and sensor pod for realism
    """
    base_length = 0.16
    base_width = 0.09
    base_height = 0.018
    
    parts = []
    
    # Base body
    base = trimesh.creation.box([base_length, base_width, base_height])
    base.apply_translation([base_length/2, base_width/2, base_height/2])
    parts.append(base)
    
    # Rounded corners on base
    fillet_radius = 0.012
    for x in [fillet_radius, base_length - fillet_radius]:
        for y in [fillet_radius, base_width - fillet_radius]:
            corner = trimesh.creation.cylinder(radius=fillet_radius, height=base_height, sections=48)
            corner.apply_translation([x, y, base_height/2])
            parts.append(corner)
    
    # Top deck
    top_deck = trimesh.creation.box([base_length - 0.02, base_width - 0.02, 0.004])
    top_deck.apply_translation([base_length/2, base_width/2, base_height + 0.002])
    parts.append(top_deck)
    
    # Wrist mounting ring
    ring = trimesh.creation.cylinder(radius=0.022, height=0.01, sections=80)
    ring.apply_translation([base_length * 0.28, base_width/2, base_height + 0.005])
    parts.append(ring)
    
    ring_flange = trimesh.creation.torus(major_radius=0.022, minor_radius=0.003)
    ring_flange.apply_translation([base_length * 0.28, base_width/2, base_height + 0.005])
    parts.append(ring_flange)
    
    # Side stiffeners
    for y in [0.012, base_width - 0.012]:
        rib = trimesh.creation.box([0.11, 0.006, 0.01])
        rib.apply_translation([base_length * 0.60, y, base_height + 0.005])
        parts.append(rib)
    
    # Precision rails
    rail_length = base_length * 0.62
    rail_height = 0.010
    rail_width = 0.012
    rail_y_positions = [0.025, base_width - 0.025]
    for y in rail_y_positions:
        rail = trimesh.creation.box([rail_length, rail_width, rail_height])
        rail.apply_translation([base_length * 0.58, y, base_height + rail_height/2 + 0.002])
        parts.append(rail)
    
    # Sliding jaws
    jaw_length = 0.028
    jaw_width = 0.022
    jaw_height = 0.018
    jaw_x = base_length * 0.80
    for y in rail_y_positions:
        jaw = trimesh.creation.box([jaw_length, jaw_width, jaw_height])
        jaw.apply_translation([jaw_x, y, base_height + jaw_height/2 + 0.003])
        parts.append(jaw)
        
        # Jaw pad with rounded face
        pad = trimesh.creation.cylinder(radius=0.008, height=jaw_width, sections=36)
        rot = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        pad.apply_transform(rot)
        pad.apply_translation([jaw_x + jaw_length/2 - 0.004, y, base_height + jaw_height/2 + 0.003])
        parts.append(pad)
    
    # Pneumatic actuator body (aligned along X)
    actuator = trimesh.creation.cylinder(radius=0.01, height=0.08, sections=72)
    rot_x = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
    actuator.apply_transform(rot_x)
    actuator.apply_translation([base_length * 0.32, base_width/2, base_height + 0.01])
    parts.append(actuator)
    
    # Actuator rod
    rod = trimesh.creation.cylinder(radius=0.005, height=0.065, sections=48)
    rod.apply_transform(rot_x)
    rod.apply_translation([base_length * 0.32 + 0.06, base_width/2, base_height + 0.01])
    parts.append(rod)
    
    # Clevis block at rod end
    clevis = trimesh.creation.box([0.015, 0.02, 0.012])
    clevis.apply_translation([base_length * 0.32 + 0.095, base_width/2, base_height + 0.01])
    parts.append(clevis)
    
    # Fastener bosses on deck
    fastener_positions = [
        (base_length * 0.20, base_width * 0.20),
        (base_length * 0.20, base_width * 0.80),
        (base_length * 0.48, base_width * 0.20),
        (base_length * 0.48, base_width * 0.80),
    ]
    for x, y in fastener_positions:
        boss = trimesh.creation.cylinder(radius=0.0035, height=0.006, sections=24)
        boss.apply_translation([x, y, base_height + 0.003])
        parts.append(boss)
    
    # Sensor pod on nose
    sensor = trimesh.creation.icosphere(subdivisions=3, radius=0.008)
    sensor.apply_translation([base_length * 0.88, base_width/2, base_height + 0.022])
    parts.append(sensor)
    
    # Cable gland detail on the side
    gland = trimesh.creation.torus(major_radius=0.012, minor_radius=0.003)
    rot_y = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
    gland.apply_transform(rot_y)
    gland.apply_translation([base_length * 0.12, base_width + 0.008, base_height * 0.8])
    parts.append(gland)
    
    result = trimesh.util.concatenate(parts)
    result.fix_normals()
    
    return result


def generate_premium_meshes(output_dir: str) -> List[str]:
    """
    Generate all premium test meshes and save to directory.
    
    Args:
        output_dir: Directory to save mesh files
        
    Returns:
        List of created file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating premium industrial meshes...")
    print("-" * 50)
    
    meshes = {
        'turbine_blade': ("Industrial Turbine Blade", create_industrial_turbine_blade),
        'pipe_assembly': ("Flanged Pipe Assembly", create_industrial_pipe_assembly),
        'precision_gear': ("Precision Involute Gear", create_precision_gear),
        'aerospace_bracket': ("Aerospace Mounting Bracket", create_aerospace_bracket),
        'robotic_gripper': ("Robotic Parallel Gripper", create_robotic_gripper),
    }
    
    created_files = []
    
    for name, (display_name, create_func) in meshes.items():
        print(f"\nCreating: {display_name}")
        
        mesh = create_func()
        
        # Clean up mesh using correct trimesh API
        mesh.process(validate=True)
        mesh.fix_normals()
        
        # Save as STL (binary for smaller size)
        stl_path = output_path / f"{name}.stl"
        mesh.export(str(stl_path))
        created_files.append(str(stl_path))
        
        print(f"  ✓ Vertices: {len(mesh.vertices):,}")
        print(f"  ✓ Faces: {len(mesh.faces):,}")
        dims = mesh.bounds[1] - mesh.bounds[0]
        print(f"  ✓ Dimensions: {dims[0]*1000:.1f} × {dims[1]*1000:.1f} × {dims[2]*1000:.1f} mm")
        print(f"  ✓ Saved: {stl_path.name}")
    
    print("-" * 50)
    print(f"Generated {len(created_files)} premium meshes")
    
    return created_files


def get_premium_defects(mesh_name: str) -> List[dict]:
    """
    Get realistic defect definitions for premium meshes.
    
    Args:
        mesh_name: Name of the mesh
        
    Returns:
        List of defect dicts with position, type, severity, normal
    """
    defects_map = {
        'turbine_blade': [
            {'position': (0.01, 0.008, 0.25), 'type': 'leading_edge_crack', 'severity': 'high', 
             'confidence': 0.94, 'normal': (0.5, 0.8, 0.2)},
            {'position': (-0.02, -0.005, 0.15), 'type': 'surface_erosion', 'severity': 'medium', 
             'confidence': 0.87, 'normal': (-0.3, -0.9, 0.2)},
            {'position': (0.0, 0.01, 0.05), 'type': 'foreign_object_damage', 'severity': 'high', 
             'confidence': 0.91, 'normal': (0.1, 0.95, 0.2)},
        ],
        'pipe_assembly': [
            {'position': (0.03, 0.0, 0.09), 'type': 'weld_crack', 'severity': 'high', 
             'confidence': 0.96, 'normal': (1.0, 0.0, 0.0)},
            {'position': (0.0, 0.08, 0.09), 'type': 'corrosion_pit', 'severity': 'medium', 
             'confidence': 0.82, 'normal': (0.0, 1.0, 0.0)},
            {'position': (-0.02, 0.02, 0.01), 'type': 'flange_face_damage', 'severity': 'low', 
             'confidence': 0.75, 'normal': (0.0, 0.0, -1.0)},
        ],
        'precision_gear': [
            {'position': (0.035, 0.0, 0.008), 'type': 'tooth_pitting', 'severity': 'high', 
             'confidence': 0.93, 'normal': (1.0, 0.0, 0.0)},
            {'position': (0.0, 0.035, 0.008), 'type': 'wear_pattern', 'severity': 'medium', 
             'confidence': 0.85, 'normal': (0.0, 1.0, 0.0)},
            {'position': (0.012, 0.0, 0.02), 'type': 'keyway_crack', 'severity': 'high', 
             'confidence': 0.89, 'normal': (0.7, 0.0, 0.7)},
        ],
        'aerospace_bracket': [
            {'position': (0.005, 0.03, 0.04), 'type': 'fatigue_crack', 'severity': 'high', 
             'confidence': 0.97, 'normal': (1.0, 0.0, 0.0)},
            {'position': (0.07, 0.03, 0.002), 'type': 'stress_corrosion', 'severity': 'medium', 
             'confidence': 0.84, 'normal': (0.0, 0.0, -1.0)},
            {'position': (0.025, 0.015, 0.004), 'type': 'fretting_wear', 'severity': 'low', 
             'confidence': 0.78, 'normal': (0.0, 0.0, 1.0)},
        ],
        'robotic_gripper': [
            {'position': (0.13, 0.024, 0.028), 'type': 'jaw_misalignment', 'severity': 'medium', 
             'confidence': 0.88, 'normal': (1.0, 0.0, 0.0)},
            {'position': (0.06, 0.045, 0.020), 'type': 'pneumatic_leak', 'severity': 'high', 
             'confidence': 0.91, 'normal': (0.0, 0.0, 1.0)},
            {'position': (0.12, 0.07, 0.022), 'type': 'sensor_offset', 'severity': 'low', 
             'confidence': 0.76, 'normal': (0.0, 1.0, 0.0)},
        ],
    }
    
    return defects_map.get(mesh_name, [])


if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else './assets/premium_meshes'
    generate_premium_meshes(output_dir)
