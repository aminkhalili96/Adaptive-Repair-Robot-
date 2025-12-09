"""
AARR Operator Control Station - Interactive 3D Repair Interface

Features:
- Interactive 3D mesh visualization with Plotly
- Premium industrial meshes (Turbine, Gear, Pipe, Bracket, Gripper)
- Gemini showcase parts (Fuselage, Pipe Bend, Saddle)
- Multi-Agent Chat (Supervisor, Inspector, Engineer)
- File upload for OBJ/STL parts
- Defect detection and repair planning workflow
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

# Add project root to path (moved here to ensure sys is available)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports from project modules
from src.visualization.mesh_loader import load_mesh, load_mesh_from_bytes, MeshData, sample_surface_points
from src.visualization.plotly_viewer import Mesh3DViewer
from src.visualization.premium_meshes import generate_premium_meshes, get_premium_defects
from src.visualization.demo_part_generator import generate_demo_part
from src.visualization.gemini_models import (
    generate_aircraft_fuselage,
    generate_complex_pipe_bend,
    generate_saddle_shape,
)
from src.agent.multi_agent_chat import MultiAgentTeam
from src.config import config

# Premium dark theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:wght@400;500;600&display=swap');
    
    .stApp {
        background: #1a1a1a !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    .main .block-container { padding: 1rem 2rem; max-width: 100%; }
    
    h1, h2, h3 {
        font-family: 'Source Serif 4', Georgia, serif !important;
        color: #ececec !important;
    }
    
    [data-testid="stSidebar"] {
        background: #252525 !important;
        border-right: 1px solid #333;
    }
    
    .stButton > button {
        background: #d97757 !important;
        color: #1a1a1a !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    
    .stButton > button:hover { background: #e8956d !important; }
    .stButton > button:disabled { background: #333 !important; color: #666 !important; }
    
    .stChatMessage { background: #252525 !important; border: 1px solid #333; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ============ SESSION STATE ============
def init_state():
    """Initialize session state with defaults."""
    defaults = {
        "mesh_data": None,
        "mesh_display_trace": None,
        "mesh_name": None,
        "mesh_source": "none",  # 'upload', 'premium', 'procedural'
        "defects": [],
        "defect_normals": [],
        "plans": [],
        "toolpath": [],
        "chat_history": [],
        "highlight_position": None,
        "camera_target": None,
        "workflow_step": 0,
        "approved": False,
        "agent_team": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
    if st.session_state.agent_team is None:
        st.session_state.agent_team = MultiAgentTeam()

init_state()


# ============ HELPER FUNCTIONS ============
def reset_state():
    """Clear all part-related state."""
    st.session_state.mesh_data = None
    st.session_state.mesh_display_trace = None
    st.session_state.mesh_name = None
    st.session_state.mesh_source = "none"
    st.session_state.defects = []
    st.session_state.defect_normals = []
    st.session_state.plans = []
    st.session_state.toolpath = []
    st.session_state.highlight_position = None
    st.session_state.camera_target = None
    st.session_state.workflow_step = 0
    st.session_state.approved = False


def load_premium_mesh(mesh_key: str):
    """Load a premium STL mesh with predefined defects."""
    mesh_dir = Path("assets/premium_meshes")
    mesh_path = mesh_dir / f"{mesh_key}.stl"
    
    if not mesh_path.exists():
        with st.spinner(f"Generating {mesh_key.replace('_', ' ').title()}..."):
            generate_premium_meshes(str(mesh_dir))
    
    mesh_data = load_mesh(str(mesh_path))
    
    reset_state()
    st.session_state.mesh_data = mesh_data
    st.session_state.mesh_name = mesh_key.replace('_', ' ').title()
    st.session_state.mesh_source = 'premium'
    st.session_state.workflow_step = 1
    
    # Load predefined defects
    defects = get_premium_defects(mesh_key)
    if defects:
        st.session_state.defects = defects
        st.session_state.defect_normals = [d.get('normal', (0, 0, 1)) for d in defects]
        st.session_state.workflow_step = 2


def load_procedural_mesh(selection: str):
    """Load a procedural/Gemini mesh."""
    reset_state()
    st.session_state.mesh_source = 'procedural'
    st.session_state.mesh_name = selection
    
    generators = {
        "Auto Body Panel": generate_demo_part,
        "Aircraft Fuselage": generate_aircraft_fuselage,
        "Complex Pipe Bend": generate_complex_pipe_bend,
        "Saddle Shape": generate_saddle_shape,
    }
    
    if selection in generators:
        st.session_state.mesh_display_trace = generators[selection]()
        st.session_state.workflow_step = 2  # Defects built-in


def load_uploaded_mesh(uploaded_file):
    """Load user-uploaded mesh."""
    reset_state()
    file_type = uploaded_file.name.split('.')[-1]
    
    with st.spinner(f"Loading {uploaded_file.name}..."):
        mesh_data = load_mesh_from_bytes(
            uploaded_file.getvalue(),
            file_type,
            name=uploaded_file.name.rsplit('.', 1)[0]
        )
    
    st.session_state.mesh_data = mesh_data
    st.session_state.mesh_name = mesh_data.name
    st.session_state.mesh_source = 'upload'
    st.session_state.workflow_step = 1


def perform_scan():
    """Simulate defect scanning."""
    if not st.session_state.mesh_data:
        return
    
    positions, normals = sample_surface_points(st.session_state.mesh_data, n_points=3)
    defect_types = ['crack', 'corrosion', 'wear', 'pitting']
    severities = ['high', 'medium', 'low']
    
    st.session_state.defects = [
        {
            'position': tuple(positions[i]),
            'type': np.random.choice(defect_types),
            'severity': np.random.choice(severities),
            'confidence': np.random.uniform(0.75, 0.98),
            'normal': tuple(normals[i]) if i < len(normals) else (0, 0, 1)
        }
        for i in range(len(positions))
    ]
    st.session_state.defect_normals = [d['normal'] for d in st.session_state.defects]
    st.session_state.workflow_step = 2


def generate_plans():
    """Generate repair plans."""
    from src.agent.tools import get_fallback_plan
    
    plans = []
    for i, defect in enumerate(st.session_state.defects):
        plan = get_fallback_plan(defect['type'])
        plans.append({
            "index": i,
            "defect_type": defect['type'],
            "position": defect['position'],
            "severity": defect['severity'],
            **plan
        })
    
    st.session_state.plans = plans
    st.session_state.workflow_step = 3


# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("## 🤖 AARR")
    st.markdown("*Operator Control Station*")
    st.markdown("---")
    
    # Part Selection
    st.markdown("### 📁 Part Selection")
    
    part_mode = st.radio(
        "Mode",
        options=["Premium Parts", "Showcase Parts", "Upload"],
        horizontal=True,
        key="part_mode"
    )
    
    if part_mode == "Premium Parts":
        premium_options = ["", "turbine_blade", "pipe_assembly", "precision_gear", 
                          "aerospace_bracket", "robotic_gripper"]
        selection = st.selectbox(
            "Load Premium Part",
            options=premium_options,
            format_func=lambda x: "Select..." if x == "" else x.replace('_', ' ').title(),
            key="premium_selector"
        )
        
        if selection and st.session_state.mesh_source != 'premium':
            load_premium_mesh(selection)
            st.rerun()
    
    elif part_mode == "Showcase Parts":
        showcase_options = ["", "Auto Body Panel", "Aircraft Fuselage", 
                          "Complex Pipe Bend", "Saddle Shape"]
        selection = st.selectbox(
            "Load Showcase Part",
            options=showcase_options,
            format_func=lambda x: "Select..." if x == "" else x,
            key="showcase_selector"
        )
        
        if selection and st.session_state.mesh_source != 'procedural':
            load_procedural_mesh(selection)
            st.rerun()
    
    else:  # Upload
        uploaded_file = st.file_uploader(
            "Upload OBJ/STL",
            type=["obj", "stl"],
            help="Upload your own 3D model"
        )
        if uploaded_file and st.session_state.mesh_source != 'upload':
            load_uploaded_mesh(uploaded_file)
            st.rerun()
    
    st.markdown("---")
    
    # Multi-Agent Chat
    st.markdown("### 💬 Factory Team")
    st.caption("🤖 Supervisor · 👁️ Inspector · 🔧 Engineer")
    
    chat_container = st.container(height=250)
    with chat_container:
        for msg in st.session_state.chat_history:
            avatar = msg.get("avatar", "👤" if msg["role"] == "user" else "🤖")
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
    
    user_input = st.chat_input("Ask the team...", key="chat_input")
    
    if user_input:
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_input,
            "avatar": "👤"
        })
        
        st.session_state.agent_team.update_state(
            defects=st.session_state.defects,
            plans=st.session_state.plans
        )
        
        response = st.session_state.agent_team.process_message(user_input)
        
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response["content"],
            "avatar": response["avatar"],
            "agent": response["agent"]
        })
        
        # Process UI commands
        for cmd in response.get("ui_commands", []):
            if cmd["type"] == "HIGHLIGHT_DEFECT" and cmd.get("position"):
                st.session_state.highlight_position = cmd["position"]
            if cmd["type"] == "ZOOM_TO" and cmd.get("position"):
                st.session_state.camera_target = cmd["position"]
        
        st.rerun()


# ============ MAIN CONTENT ============
st.markdown("""
<div style="margin-bottom: 16px;">
    <h1 style="font-size: 24px; margin: 0;">Operator Control Station</h1>
    <p style="color: #666; margin-top: 4px; font-size: 13px;">Interactive 3D defect inspection and repair planning</p>
</div>
""", unsafe_allow_html=True)

col_viewer, col_actions = st.columns([2.5, 1])

with col_viewer:
    # Procedural mesh (Mesh3d trace)
    if st.session_state.mesh_display_trace:
        fig = go.Figure(data=[st.session_state.mesh_display_trace])
        fig.update_layout(
            scene=dict(aspectmode='data', bgcolor='#1a1a1a'),
            paper_bgcolor='#1a1a1a',
            height=550,
            margin=dict(l=0, r=0, b=0, t=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        <div style="background: #252525; padding: 12px 16px; border-radius: 8px;">
            <span style="color: #d97757; font-weight: 600;">✨ Showcase Part</span>
            <span style="color: #888; margin-left: 12px;">{st.session_state.mesh_name}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Standard mesh (MeshData)
    elif st.session_state.mesh_data:
        viewer = Mesh3DViewer(st.session_state.mesh_data)
        
        if st.session_state.defects:
            positions = [d['position'] for d in st.session_state.defects]
            labels = [d['type'].capitalize() for d in st.session_state.defects]
            severities = [d['severity'] for d in st.session_state.defects]
            normals = st.session_state.defect_normals or None
            confidences = [d['confidence'] for d in st.session_state.defects]
            
            viewer.add_defect_markers(positions, labels, severities, normals, confidences)
        
        if st.session_state.toolpath:
            viewer.add_toolpath(st.session_state.toolpath)
        
        if st.session_state.highlight_position:
            viewer.highlight_region(st.session_state.highlight_position, radius=0.015)
        
        fig = viewer.create_figure()
        
        if st.session_state.camera_target:
            camera = viewer.set_camera_view(st.session_state.camera_target)
            fig.update_layout(scene_camera=camera)
        
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        
        # Part info
        mesh = st.session_state.mesh_data
        dims = mesh.bounds[1] - mesh.bounds[0]
        st.markdown(f"""
        <div style="display: flex; gap: 24px; padding: 12px 0; border-top: 1px solid #333; color: #888; font-size: 12px;">
            <span><strong>Part:</strong> {st.session_state.mesh_name}</span>
            <span><strong>Vertices:</strong> {len(mesh.vertices):,}</span>
            <span><strong>Dimensions:</strong> {dims[0]*1000:.1f} × {dims[1]*1000:.1f} × {dims[2]*1000:.1f} mm</span>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div style="background: #252525; border: 1px dashed #444; border-radius: 12px; 
             height: 500px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <span style="font-size: 48px; margin-bottom: 16px;">🔧</span>
            <span style="color: #888; font-size: 16px;">No Part Loaded</span>
            <span style="color: #666; font-size: 13px; margin-top: 8px;">Select a part from the sidebar</span>
        </div>
        """, unsafe_allow_html=True)


with col_actions:
    st.markdown("#### Workflow")
    
    step = st.session_state.workflow_step
    steps = ["Load Part", "Scan", "Plan", "Approve", "Execute"]
    
    for i, name in enumerate(steps):
        if i < step:
            icon, color = "✓", "#7dca9a"
        elif i == step:
            icon, color = "●", "#d97757"
        else:
            icon, color = str(i+1), "#666"
        st.markdown(f'<span style="color: {color}; font-weight: 600;">{icon}</span> <span style="color: {color};">{name}</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### Actions")
    
    # Scan - only for uploaded/premium (not procedural)
    can_scan = st.session_state.mesh_source in ['upload', 'premium'] and step == 1
    if st.button("🔍 Scan Part", disabled=not can_scan, use_container_width=True):
        with st.spinner("Scanning..."):
            perform_scan()
        st.rerun()
    
    # Plan
    if st.button("📋 Generate Plan", disabled=step != 2, use_container_width=True):
        with st.spinner("Planning..."):
            generate_plans()
        st.rerun()
    
    # Approve
    if step == 3:
        if st.checkbox("✅ Approve Plan", value=st.session_state.approved):
            st.session_state.approved = True
            st.session_state.workflow_step = 4
            st.rerun()
    
    # Execute
    if st.button("🚀 Execute", disabled=step != 4, use_container_width=True):
        st.session_state.workflow_step = 5
        st.rerun()
    
    if step >= 5:
        st.success("✓ Repair Complete")
    
    st.markdown("---")
    
    # Defects list
    st.markdown("#### Defects")
    if st.session_state.mesh_source == 'procedural':
        st.caption("Defects built into mesh")
    elif st.session_state.defects:
        for i, d in enumerate(st.session_state.defects):
            color = {'high': '#e85d5d', 'medium': '#e8a55d', 'low': '#7dca9a'}.get(d['severity'], '#888')
            if st.button(f"{d['type'].capitalize()}", key=f"def_{i}", use_container_width=True):
                st.session_state.highlight_position = d['position']
                st.session_state.camera_target = d['position']
                st.rerun()
            st.markdown(f'<span style="color: {color}; font-size: 11px;">{d["severity"].upper()}</span>', unsafe_allow_html=True)
    else:
        st.caption("No defects detected")
    
    # Reset
    if step > 0:
        st.markdown("---")
        if st.button("↻ Reset", use_container_width=True):
            reset_state()
            st.session_state.chat_history = []
            st.rerun()