"""
AARR Operator Control Station - Industrial Grade UI

Features:
- Split-view layout (70% 3D Viewer / 30% AI Agent Panel)
- Dark Industrial Theme (Fusion 360 inspired)
- Real-time status metrics bar
- Safety Orange accent for critical actions
"""

import streamlit as st
import numpy as np
import sys
import base64
import tempfile
from pathlib import Path
import plotly.graph_objects as go
from openai import OpenAI

# Page config - must be first
st.set_page_config(
    page_title="AARR Control Station",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add project root to path
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
from src.visualization.premium_procedural_models import (
    generate_premium_pipe,
    generate_turbine_blade_v2,
    generate_car_hood_v2,
)
from src.agent.supervisor_agent import ConversationalTeam
from src.config import config

# Import ML predictor for chart
try:
    from src.ml import get_predictor
    HAS_ML_PREDICTOR = True
except ImportError:
    HAS_ML_PREDICTOR = False

# Import SAM segmentor for interactive segmentation
try:
    from src.vision.sam_segmentor import get_segmentor, SAMSegmentor
    HAS_SAM = True
except ImportError:
    HAS_SAM = False

# ============ CLAUDE AESTHETIC THEME ============
# Warm paper-like design inspired by Anthropic's Claude
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Inter:wght@400;500;600&display=swap');
    
    /* === GLOBAL - COMPACT WARM PAPER AESTHETIC === */
    .stApp {
        background: #FDFBF9 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header, .stDeployButton {display: none !important; visibility: hidden !important;}
    
    /* COMPACT padding */
    .main .block-container {
        padding: 0.5rem 1rem !important;
        max-width: 100% !important;
    }
    
    /* === SIDEBAR - WARM BEIGE === */
    [data-testid="stSidebar"] {
        background: #F4F1EA !important;
        border-right: 1px solid #E8E4DC !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #2D2D2D !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        font-family: 'Merriweather', Georgia, serif !important;
        color: #1A1A1A !important;
        font-weight: 700 !important;
    }
    
    /* === TYPOGRAPHY - SERIF HEADERS === */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Merriweather', Georgia, serif !important;
        color: #1A1A1A !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }
    
    p, span, label, .stMarkdown, div {
        color: #2D2D2D !important;
    }
    
    /* === BUTTONS - COMPACT BURNT ORANGE === */
    .stButton > button[kind="primary"], 
    .stButton > button {
        background: linear-gradient(135deg, #D97757 0%, #C4654A 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 6px 14px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 4px rgba(217, 119, 87, 0.2) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #E08A6D 0%, #D97757 100%) !important;
        box-shadow: 0 4px 16px rgba(217, 119, 87, 0.35) !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton > button:disabled {
        background: #E8E4DC !important;
        color: #9A9A9A !important;
        box-shadow: none !important;
    }
    
    /* Secondary buttons */
    .stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #2D2D2D !important;
        border: 1px solid #E0E0E0 !important;
        box-shadow: none !important;
    }
    
    /* === METRICS BAR - LIGHT CARD === */
    .metrics-bar {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-bottom: 2px solid #F3F4F6;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 32px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    .metric-item {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .metric-label {
        color: #6B6B6B !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .metric-value {
        color: #1A1A1A !important;
        font-size: 14px !important;
        font-weight: 600;
    }
    
    .status-online {
        color: #2E7D32 !important;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #2E7D32;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* === AGENT PANEL - DOCUMENT STYLE === */
    .agent-panel {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        height: 100%;
        display: flex;
        flex-direction: column;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    .agent-header {
        background: #FAFAFA;
        padding: 16px 20px;
        border-bottom: 1px solid #E0E0E0;
        border-radius: 12px 12px 0 0;
    }
    
    .agent-title {
        font-family: 'Merriweather', serif !important;
        color: #1A1A1A !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        margin: 0 !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .agent-subtitle {
        color: #6B6B6B !important;
        font-size: 12px !important;
        margin-top: 4px !important;
    }
    
    /* Chat messages - ChatGPT style floating cards */
    .chat-message {
        padding: 16px 20px;
        margin: 12px 16px;
        border-radius: 16px;
        max-width: 85%;
    }
    
    .chat-user {
        background: #F3F4F6 !important;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .chat-agent {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    
    .chat-agent-icon {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: #D97757;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        color: #FFFFFF;
        margin-right: 8px;
    }
    
    /* === NATIVE st.chat_message STYLING === */
    /* User messages - accent color pop on glass */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse !important;
        background: linear-gradient(135deg, #D97757 0%, #C4654A 100%) !important;
        border-radius: 16px !important;
        border-bottom-right-radius: 4px !important;
        margin: 8px 0 !important;
        padding: 12px 16px !important;
        box-shadow: 0 2px 8px rgba(217, 119, 87, 0.25) !important;
    }
    
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) span {
        color: #FFFFFF !important;
    }
    
    /* AI messages - clean white on glass */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #E5E7EB !important;
        box-shadow: none !important;
        border-radius: 16px !important;
        border-bottom-left-radius: 4px !important;
        margin: 8px 0 !important;
        padding: 12px 16px !important;
    }
    
    [data-testid="stChatMessage"] {
        border: none !important;
    }
    
    /* === GLASSMORPHISM - RIGHT COLUMN (CHAT PANEL) === */
    /* Target the second column's content */
    [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-of-type(2) > div {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05) !important;
        padding: 20px !important;
        min-height: 70vh !important;
    }
    
    /* Glass panel header refinement */
    .agent-header {
        background: rgba(255, 255, 255, 0.5) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: none !important;
        border-bottom: 1px solid rgba(224, 224, 224, 0.5) !important;
        border-radius: 12px 12px 0 0 !important;
        margin: -20px -20px 16px -20px !important;
        padding: 16px 20px !important;
    }
    
    /* Chat input on glass */
    .stChatInput > div {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(224, 224, 224, 0.6) !important;
        border-radius: 12px !important;
    }
    
    /* === AUDIO INPUT - HIDE TIMELINE === */
    [data-testid="stAudioInput"] {
        max-width: 200px !important;
    }
    
    [data-testid="stAudioInput"] audio::-webkit-media-controls-timeline,
    [data-testid="stAudioInput"] audio::-webkit-media-controls-time-remaining-display,
    [data-testid="stAudioInput"] audio::-webkit-media-controls-current-time-display {
        display: none !important;
    }
    
    /* Firefox audio controls */
    [data-testid="stAudioInput"] audio {
        width: 80px !important;
    }
    
    /* === VIEWPORT CARD - INDUSTRIAL DASHBOARD === */
    .viewport-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        padding: 0;
        overflow: hidden;
        margin-bottom: 16px;
    }
    
    .viewport-hud {
        background: #FAFAFA;
        border-bottom: 1px solid #E5E7EB;
        padding: 10px 16px;
        display: flex;
        align-items: center;
        gap: 24px;
        font-size: 12px;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    .hud-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .hud-label {
        color: #6B7280 !important;
        font-weight: 500;
        font-size: 11px;
    }
    
    .hud-value {
        color: #1F2937 !important;
        font-weight: 600;
        font-size: 12px;
    }
    
    .status-dot-live {
        width: 8px;
        height: 8px;
        background: #10B981;
        border-radius: 50%;
        animation: pulse-dot 2s infinite;
    }
    
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(0.9); }
    }
    
    /* === 3D VIEWER CONTAINER - TRANSPARENT === */
    .viewer-container {
        background: transparent;
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        padding: 0;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    .viewer-toolbar {
        background: #FAFAFA;
        padding: 12px 16px;
        border-bottom: 1px solid #E0E0E0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* === WORKFLOW STEPS === */
    .workflow-step {
        display: flex;
        align-items: center;
        padding: 8px 0;
        gap: 12px;
    }
    
    .step-icon {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 600;
    }
    
    .step-complete {
        background: #E8F5E9;
        color: #2E7D32;
    }
    
    .step-active {
        background: #D97757;
        color: #FFFFFF;
    }
    
    .step-pending {
        background: #F4F1EA;
        color: #9A9A9A;
    }
    
    /* === DEFECT CARDS === */
    .defect-card {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .defect-card:hover {
        border-color: #D97757;
        box-shadow: 0 2px 8px rgba(217, 119, 87, 0.15);
    }
    
    .severity-high { border-left: 3px solid #C62828 !important; }
    .severity-medium { border-left: 3px solid #EF6C00 !important; }
    .severity-low { border-left: 3px solid #2E7D32 !important; }
    
    /* === SCROLLBAR - SUBTLE === */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F4F1EA;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #D0CBC3;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #B8B3AB;
    }
    
    /* === SELECT BOX & DROPDOWN === */
    .stSelectbox > div > div {
        background: #FFFFFF !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
        color: #2D2D2D !important;
    }
    
    /* Dropdown menu list (the popup) */
    .stSelectbox [data-baseweb="popover"],
    .stSelectbox [data-baseweb="menu"],
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
    }
    
    .stSelectbox [data-baseweb="menu"] li,
    [data-baseweb="menu"] li {
        background: #FFFFFF !important;
        color: #2D2D2D !important;
    }
    
    .stSelectbox [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] li:hover {
        background: #F4F1EA !important;
    }
    
    /* === FILE UPLOADER === */
    [data-testid="stFileUploader"] {
        background: #FFFFFF !important;
        border: 1px dashed #D0CBC3 !important;
        border-radius: 8px !important;
        padding: 16px !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #D97757 !important;
    }
    
    /* === CHAT INPUT === */
    .stChatInput > div {
        background: #FFFFFF !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
    }
    
    .stChatInput input {
        color: #2D2D2D !important;
    }
    
    /* === RADIO BUTTONS === */
    .stRadio > div {
        gap: 8px !important;
    }
    
    .stRadio label {
        background: #FFFFFF !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        color: #2D2D2D !important;
    }
    
    .stRadio label[data-checked="true"] {
        background: #D97757 !important;
        border-color: #D97757 !important;
        color: #FFFFFF !important;
    }
    
    /* === EXPANDER === */
    .streamlit-expanderHeader {
        background: #FAFAFA !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
    }
    
    /* === SUCCESS/INFO/WARNING BOXES === */
    .stSuccess {
        background: #E8F5E9 !important;
        color: #1B5E20 !important;
    }
    
    .stInfo {
        background: #FFF8E1 !important;
        color: #E65100 !important;
    }
    
    /* === VOICE MIC BUTTON - COMPACT 44px === */
    [data-testid="stAudioInput"] {
        width: 48px !important;
        max-width: 48px !important;
        min-width: 48px !important;
        height: 48px !important;
        overflow: hidden !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #D97757 0%, #C4654A 100%) !important;
        padding: 0 !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(217, 119, 87, 0.35) !important;
    }
    
    [data-testid="stAudioInput"] > div {
        width: 48px !important;
        height: 48px !important;
        padding: 0 !important;
    }
    
    /* Hide all audio player elements except record button */
    [data-testid="stAudioInput"] audio,
    [data-testid="stAudioInput"] canvas,
    [data-testid="stAudioInput"] .waveform,
    [data-testid="stAudioInput"] [data-testid*="time"],
    [data-testid="stAudioInput"] span:not(:has(button)),
    [data-testid="stAudioInput"] label {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Style the record button as circle */
    [data-testid="stAudioInput"] button {
        width: 48px !important;
        height: 48px !important;
        min-width: 48px !important;
        border-radius: 50% !important;
        background: transparent !important;
        border: none !important;
        color: #FFFFFF !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    [data-testid="stAudioInput"] button:hover {
        transform: scale(1.08) !important;
        background: rgba(255,255,255,0.1) !important;
    }
    
    [data-testid="stAudioInput"] button svg {
        fill: #FFFFFF !important;
        width: 20px !important;
        height: 20px !important;
    }
    
    /* === POPOVER MIC BUTTON - CLEAN SQUARE === */
    [data-testid="stPopover"] > button {
        width: 45px !important;
        height: 45px !important;
        min-width: 45px !important;
        padding: 0 !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
        background: #FFFFFF !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 20px !important;
    }
    
    /* Hide the caret/dropdown arrow in popover button - AGGRESSIVE */
    [data-testid="stPopover"] > button svg,
    [data-testid="stPopover"] > button [data-testid="stIconMaterial"],
    [data-testid="stPopover"] > button > div:last-child,
    [data-testid="stPopover"] > button > span > svg {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
    }
    
    /* Only show the emoji text */
    [data-testid="stPopover"] > button > span:first-child {
        font-size: 20px !important;
    }
    
    [data-testid="stPopover"] > button:hover {
        background: #F9FAFB !important;
        border-color: #D97757 !important;
    }
    
    /* === CHAT LAYOUT - TIGHTER SPACING === */
    /* Reduce gap between chat history and input */
    [data-testid="stVerticalBlockBorderWrapper"] {
        gap: 8px !important;
    }
    
    /* Chat input container - compact */
    .stChatInput {
        margin-top: 8px !important;
        padding-top: 0 !important;
    }
    
    /* Chat message container - tighter */
    [data-testid="stChatMessageContent"] {
        padding: 8px 12px !important;
    }
    
    /* === VIEWPORT TOOLBAR - ANCHORED HEADER === */
    .viewport-toolbar-anchored {
        background: #F9FAFB;
        padding: 12px 16px;
        border: 1px solid #E5E7EB;
        border-bottom: 2px solid #E5E7EB;
        border-radius: 12px 12px 0 0;
        display: flex;
        align-items: center;
        gap: 16px;
        font-family: 'Inter', -apple-system, sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# ============ SAMPLE PARTS REGISTRY ============
SAMPLE_PARTS = {
    # Premium STL Parts (from file)
    "turbine_blade": {"type": "premium", "name": "Turbine Blade", "desc": "Industrial airfoil"},
    "pipe_assembly": {"type": "premium", "name": "Pipe Assembly", "desc": "Flanged pipe"},
    "precision_gear": {"type": "premium", "name": "Precision Gear", "desc": "Involute gear"},
    "aerospace_bracket": {"type": "premium", "name": "Aerospace Bracket", "desc": "Lightened mount"},
    "robotic_gripper": {"type": "premium", "name": "Robotic Gripper", "desc": "Parallel gripper"},
    # High-Fidelity Procedural Parts (new!)
    "flanged_pipe": {"type": "procedural", "name": "Flanged Pipe", "desc": "Industrial pipe with flanges"},
    "naca_blade": {"type": "procedural", "name": "NACA Turbine Blade", "desc": "Twisted airfoil with heat stress"},
    "car_hood": {"type": "procedural", "name": "Car Hood Panel", "desc": "Solid panel with rust"},
    # Legacy Procedural Parts
    "aircraft_fuselage": {"type": "procedural", "name": "Aircraft Fuselage", "desc": "Cylindrical section"},
    "pipe_bend": {"type": "procedural", "name": "Complex Pipe Bend", "desc": "Pipe elbow"},
    "saddle_shape": {"type": "procedural", "name": "Saddle Shape", "desc": "Hyperbolic surface"},
    # Sci-Fi Robot (new!)
    "scifi_drone": {"type": "robot", "name": "Sci-Fi Scout Drone", "desc": "Articulated robot with legs"},
}


# ============ SESSION STATE ============
def init_state():
    defaults = {
        "mesh_data": None,
        "mesh_display_trace": None,
        "mesh_name": "No Part Loaded",
        "mesh_source": "none",
        "defects": [],
        "defect_normals": [],
        "plans": [],
        "toolpath": [],
        "chat_history": [],
        "highlight_position": None,
        "camera_target": None,
        "camera_eye": dict(x=1.5, y=1.5, z=1.5),  # Direct camera eye control
        "workflow_step": 0,
        "approved": False,
        "agent_team": None,
        "current_part_key": None,
        "pending_visual_request": False,  # Flag for visual analysis pending
        "current_figure": None,  # Store current Plotly figure for snapshot
        # SAM Interactive Segmentation
        "sam_enabled": False,
        "sam_snapshot": None,  # Captured image for segmentation
        "sam_result": None,  # SegmentationResult from SAM
        "sam_click_x": 400,  # Default click X coordinate
        "sam_click_y": 300,  # Default click Y coordinate
        # Custom Path (Code Interpreter)
        "custom_path_points": None,  # Generated path points from LLM code
        "custom_path_info": None,  # Pattern description and metadata
        # Simulation video recording
        "simulation_video": None,  # Path to recorded simulation video
        # Quality Assurance Report
        "qa_report": None,  # Cached QA report data
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
    if st.session_state.agent_team is None:
        st.session_state.agent_team = ConversationalTeam()

init_state()


# ============ HELPER FUNCTIONS ============
def reset_state():
    st.session_state.mesh_data = None
    st.session_state.mesh_display_trace = None
    st.session_state.mesh_name = "No Part Loaded"
    st.session_state.mesh_source = "none"
    st.session_state.defects = []
    st.session_state.defect_normals = []
    st.session_state.plans = []
    st.session_state.toolpath = []
    st.session_state.highlight_position = None
    st.session_state.camera_target = None
    st.session_state.workflow_step = 0
    st.session_state.approved = False
    st.session_state.current_part_key = None
    st.session_state.pending_visual_request = False
    st.session_state.current_figure = None
    st.session_state.sam_enabled = False
    st.session_state.sam_snapshot = None
    st.session_state.sam_result = None
    st.session_state.sam_click_x = 400
    st.session_state.sam_click_y = 300
    st.session_state.custom_path_points = None
    st.session_state.custom_path_info = None
    st.session_state.simulation_video = None
    st.session_state.qa_report = None


def capture_figure_as_base64(fig: go.Figure) -> str:
    """Capture a Plotly figure as a base64-encoded PNG using Kaleido."""
    try:
        img_bytes = fig.to_image(format="png", width=800, height=600)
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error capturing figure: {e}")
        return None


def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    try:
        client = OpenAI()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            f.flush()
            with open(f.name, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
        return transcript.text
    except Exception as e:
        return f"Error: {str(e)}"


def load_sample_part(part_key: str):
    if part_key not in SAMPLE_PARTS:
        return
    
    part_info = SAMPLE_PARTS[part_key]
    reset_state()
    st.session_state.current_part_key = part_key
    st.session_state.mesh_name = part_info["name"]
    
    if part_info["type"] == "premium":
        mesh_dir = Path("assets/premium_meshes")
        mesh_path = mesh_dir / f"{part_key}.stl"
        
        if not mesh_path.exists():
            generate_premium_meshes(str(mesh_dir))
        
        st.session_state.mesh_data = load_mesh(str(mesh_path))
        st.session_state.mesh_source = "premium"
        # Don't auto-load defects - user must click SCAN PART first
        st.session_state.workflow_step = 1
    elif part_info["type"] == "robot":
        # Sci-Fi Drone robot - special handling
        from src.visualization.robot_generator import get_robot_trace
        st.session_state.mesh_display_trace = None  # Reset
        st.session_state.robot_traces = get_robot_trace()  # Store multiple traces
        st.session_state.mesh_source = "robot"
        st.session_state.workflow_step = 1
    else:
        generators = {
            # High-Fidelity Models (new)
            "flanged_pipe": generate_premium_pipe,
            "naca_blade": generate_turbine_blade_v2,
            "car_hood": generate_car_hood_v2,
            # Legacy Models
            "auto_body_panel": generate_demo_part,
            "aircraft_fuselage": generate_aircraft_fuselage,
            "pipe_bend": generate_complex_pipe_bend,
            "saddle_shape": generate_saddle_shape,
        }
        if part_key in generators:
            st.session_state.mesh_display_trace = generators[part_key]()
            st.session_state.mesh_source = "procedural"
            # Procedural parts also need scan first
            st.session_state.workflow_step = 1


def load_uploaded_mesh(uploaded_file):
    reset_state()
    file_type = uploaded_file.name.split('.')[-1]
    mesh_data = load_mesh_from_bytes(
        uploaded_file.getvalue(),
        file_type,
        name=uploaded_file.name.rsplit('.', 1)[0]
    )
    st.session_state.mesh_data = mesh_data
    st.session_state.mesh_name = mesh_data.name
    st.session_state.mesh_source = "upload"
    st.session_state.workflow_step = 1


def perform_scan():
    """Perform defect scan on the loaded part."""
    # For premium parts, use predefined defects
    if st.session_state.mesh_source == "premium" and st.session_state.current_part_key:
        defects = get_premium_defects(st.session_state.current_part_key)
        if defects:
            st.session_state.defects = defects
            st.session_state.defect_normals = [d.get('normal', (0, 0, 1)) for d in defects]
            st.session_state.workflow_step = 2
            return
    
    # For procedural parts, generate synthetic defects based on traces
    if st.session_state.mesh_source == "procedural":
        # Generate random defects for procedural parts
        defect_types = ['crack', 'corrosion', 'wear', 'pitting']
        severities = ['high', 'medium', 'low']
        n_defects = np.random.randint(2, 5)
        st.session_state.defects = [
            {
                'position': (np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(0, 0.3)),
                'type': np.random.choice(defect_types),
                'severity': np.random.choice(severities),
                'confidence': np.random.uniform(0.75, 0.98),
                'normal': (0, 0, 1)
            }
            for _ in range(n_defects)
        ]
        st.session_state.defect_normals = [d['normal'] for d in st.session_state.defects]
        st.session_state.workflow_step = 2
        return
    
    # For uploaded meshes, sample surface points
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
    from src.agent.tools import get_fallback_plan
    plans = []
    for i, defect in enumerate(st.session_state.defects):
        plan = get_fallback_plan(defect['type'])
        plans.append({"index": i, "defect_type": defect['type'], **plan})
    st.session_state.plans = plans
    st.session_state.workflow_step = 3


def process_user_message(user_message: str):
    """Process a user message through the agent and handle response."""
    # Add to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_message,
        "avatar": "User"
    })
    
    # Update agent state and process message
    st.session_state.agent_team.update_state(
        defects=st.session_state.defects,
        plans=st.session_state.plans,
        workflow_step=st.session_state.workflow_step,
        mesh_name=st.session_state.mesh_name
    )
    
    response = st.session_state.agent_team.process_message(user_message)
    
    # Remove emoji from avatar if present
    avatar_text = response["avatar"]
    if avatar_text == "🤖": avatar_text = "AI"
    
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response["content"],
        "avatar": avatar_text,
        "agent": response["agent"]
    })
    
    # Handle UI commands from supervisor
    for cmd in response.get("ui_commands", []):
        cmd_type = cmd.get("type", "")
        
        # Camera focus/zoom
        if cmd_type in ["FOCUS_CAMERA", "ZOOM_TO", "HIGHLIGHT_DEFECT"] and cmd.get("position"):
            st.session_state.highlight_position = cmd["position"]
            st.session_state.camera_target = cmd["position"]
        
        # Highlight specific defect
        if cmd_type == "HIGHLIGHT" and cmd.get("defect_index") is not None:
            idx = cmd["defect_index"]
            if idx < len(st.session_state.defects):
                st.session_state.highlight_position = st.session_state.defects[idx]["position"]
                st.session_state.camera_target = st.session_state.defects[idx]["position"]
        
        # Reset view
        if cmd_type == "RESET_VIEW":
            st.session_state.highlight_position = None
            st.session_state.camera_target = None
        
        # Trigger scan
        if cmd_type == "TRIGGER_SCAN":
            if st.session_state.mesh_source in ['upload', 'premium']:
                perform_scan()
        
        # Trigger plan
        if cmd_type == "TRIGGER_PLAN":
            if st.session_state.defects:
                generate_plans()
        
        # Execute repair
        if cmd_type == "EXECUTE":
            if st.session_state.plans and st.session_state.approved:
                st.session_state.workflow_step = 5
        
        # Custom path generated by Code Interpreter
        if cmd_type == "CUSTOM_PATH_GENERATED":
            data = cmd.get("data", {})
            if data.get("points"):
                st.session_state.custom_path_points = data["points"]
                st.session_state.custom_path_info = {
                    "pattern": data.get("pattern", "Custom"),
                    "num_points": data.get("num_points", len(data["points"]))
                }
        
        # Visual inspection - capture snapshot and analyze
        if cmd_type == "CAPTURE_SNAPSHOT":
            if st.session_state.current_figure is not None:
                # Capture the current figure as base64 image
                image_base64 = capture_figure_as_base64(st.session_state.current_figure)
                if image_base64:
                    # Call agent again with the image for visual analysis
                    st.session_state.agent_team.update_state(
                        defects=st.session_state.defects,
                        plans=st.session_state.plans,
                        workflow_step=st.session_state.workflow_step,
                        mesh_name=st.session_state.mesh_name
                    )
                    visual_response = st.session_state.agent_team.agent.process_message(
                        message="Analyze this image",
                        defects=st.session_state.defects,
                        plans=st.session_state.plans,
                        workflow_step=st.session_state.workflow_step,
                        mesh_name=st.session_state.mesh_name,
                        image_base64=image_base64
                    )
                    # Replace the "capturing..." message with actual analysis
                    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
                        st.session_state.chat_history[-1] = {
                            "role": "assistant",
                            "content": visual_response["content"],
                            "avatar": visual_response["avatar"],
                            "agent": visual_response["agent"]
                        }
                else:
                    # Update last message with error
                    if st.session_state.chat_history:
                        st.session_state.chat_history[-1]["content"] = "Could not capture the 3D view. Please ensure a part is loaded."
            else:
                # No figure available
                if st.session_state.chat_history:
                    st.session_state.chat_history[-1]["content"] = "No part loaded to analyze. Please load a part first."


# ============ SIDEBAR (Compact Part Selection) ============
with st.sidebar:
    st.markdown("### AARR")
    st.caption("Agentic Adaptive Repair Robot")
    st.markdown("---")
    
    st.markdown("##### Part Selection")
    
    part_mode = st.radio("Mode", ["Sample Parts", "Upload"], horizontal=True, key="part_mode", label_visibility="collapsed")
    
    if part_mode == "Sample Parts":
        options = [""] + list(SAMPLE_PARTS.keys())
        selection = st.selectbox(
            "Select Part",
            options=options,
            format_func=lambda x: "Select..." if x == "" else SAMPLE_PARTS.get(x, {}).get("name", x),
            key="sample_selector",
            label_visibility="collapsed"
        )
        if selection and selection != st.session_state.current_part_key:
            load_sample_part(selection)
            st.rerun()
    else:
        uploaded_file = st.file_uploader("Upload", type=["obj", "stl"], label_visibility="collapsed")
        if uploaded_file and st.session_state.mesh_source != "upload":
            load_uploaded_mesh(uploaded_file)
            st.rerun()
    
    st.markdown("---")
    
    # Workflow Actions
    st.markdown("##### Workflow")
    step = st.session_state.workflow_step
    
    can_scan = st.session_state.mesh_source in ['upload', 'premium', 'procedural'] and step == 1
    if st.button("SCAN PART", disabled=not can_scan, use_container_width=True):
        with st.status("Factory Team Working...", expanded=True) as status:
            import time
            st.write("**Inspector** is analyzing surface texture...")
            time.sleep(0.5)  # Simulate processing
            
            st.write("**Vision System** capturing defect regions...")
            time.sleep(0.3)
            
            st.write("**AI Model** classifying defect types...")
            time.sleep(0.4)
            
            # Actually perform the scan
            perform_scan()
            
            st.write(f"Found **{len(st.session_state.defects)}** defects")
            status.update(label="Scan Complete", state="complete", expanded=False)
            time.sleep(0.3)
        st.rerun()
    
    if st.button("GENERATE PLAN", disabled=step != 2, use_container_width=True):
        with st.status("Factory Team Planning...", expanded=True) as status:
            import time
            st.write("**Supervisor** is consulting SOPs...")
            time.sleep(0.4)
            
            st.write("**Engineer** is analyzing repair strategies...")
            time.sleep(0.3)
            
            st.write("**Path Planner** generating toolpaths...")
            time.sleep(0.4)
            
            st.write("**Optimizer** minimizing travel time...")
            time.sleep(0.3)
            
            # Actually generate the plans
            generate_plans()
            
            st.write(f"Generated **{len(st.session_state.plans)}** repair plans")
            status.update(label="Planning Complete", state="complete", expanded=False)
            time.sleep(0.3)
        st.rerun()
    
    if step == 3 and st.checkbox("Approve Plan", value=st.session_state.approved):
        st.session_state.approved = True
        st.session_state.workflow_step = 4
        st.rerun()
    
    # Show ML Prediction chart when plans exist
    if step >= 3 and st.session_state.plans and st.session_state.defects:
        if HAS_ML_PREDICTOR:
            st.markdown("---")
            st.markdown("##### ML Predictions")
            
            # Get predictor and compute chart data
            predictor = get_predictor()
            chart_data = predictor.get_actual_vs_predicted_data(
                st.session_state.defects,
                st.session_state.plans
            )
            
            # Create bar chart comparing predicted vs estimated
            fig = go.Figure()
            
            # Predicted times (ML)
            fig.add_trace(go.Bar(
                name='ML Predicted',
                x=chart_data['labels'],
                y=chart_data['predicted_times'],
                marker_color='#FF5722',
                text=[f"{t:.0f}s" for t in chart_data['predicted_times']],
                textposition='outside',
            ))
            
            # Estimated times (rule-based)
            fig.add_trace(go.Bar(
                name='Rule Estimated',
                x=chart_data['labels'],
                y=chart_data['estimated_times'],
                marker_color='#4CAF50',
                text=[f"{t:.0f}s" for t in chart_data['estimated_times']],
                textposition='outside',
            ))
            
            fig.update_layout(
                title=dict(
                    text='Predicted vs Estimated Time',
                    font=dict(size=12, color='#E0E0E0')
                ),
                barmode='group',
                height=220,
                margin=dict(l=10, r=10, t=35, b=10),
                paper_bgcolor='rgba(0,0,0,0)',  # ProAI: transparent
                plot_bgcolor='rgba(0,0,0,0)',   # ProAI: transparent
                font=dict(size=9, color='#6B6B6B'),
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                ),
                yaxis=dict(
                    title='Time (s)',
                    gridcolor='#E0E0E0',  # Light grid for ProAI
                    showgrid=True,
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5,
                    font=dict(size=9)
                ),
                showlegend=True,
            )
            
            st.plotly_chart(fig, use_container_width=True, key="ml_prediction_chart")
            
            # Show totals
            total_predicted = sum(chart_data['predicted_times'])
            total_estimated = sum(chart_data['estimated_times'])
            st.caption(f"ML Total: **{total_predicted:.0f}s** | Rule Total: **{total_estimated:.0f}s**")
    
    if st.button("EXECUTE REPAIR", disabled=step != 4, use_container_width=True):
        # Run headless simulation with video recording
        with st.spinner("Robot is executing repair..."):
            try:
                from src.simulation import create_environment, run_simulation_with_recording
                from pathlib import Path
                import time
                
                # Ensure temp directory exists
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)
                
                # Create headless simulation environment
                env = create_environment(gui=False)
                
                # Calculate steps based on number of defects (60 steps per defect)
                num_defects = len(st.session_state.defects)
                num_steps = max(120, num_defects * 60)  # At least 120 steps
                
                # Run simulation and record video
                video_path = run_simulation_with_recording(
                    env=env,
                    num_steps=num_steps,
                    output_filename="simulation_result.mp4",
                    frame_skip=4  # Record every 4th frame for smaller file
                )
                
                # Cleanup
                env.close()
                
                # Store video path in session state
                if video_path:
                    st.session_state.simulation_video = video_path
                    
            except Exception as e:
                st.error(f"Simulation error: {e}")
                st.session_state.simulation_video = None
        
        st.session_state.workflow_step = 5
        st.rerun()
    
    if step >= 5:
        st.success("REPAIR COMPLETE")
        
        # ============ QUALITY ASSURANCE REPORT CARD ============
        st.markdown("---")
        st.markdown("##### Quality Assurance Report")
        
        # Generate report data (or use cached)
        if "qa_report" not in st.session_state or st.session_state.qa_report is None:
            import uuid
            import random
            
            # Generate report once and cache it
            num_defects = len(st.session_state.defects)
            cycle_time = round(random.uniform(2.5, 6.5), 1)
            
            # Determine primary tool based on defect types
            defect_types = [d.get('type', 'unknown') for d in st.session_state.defects]
            tool_map = {
                'crack': 'Welding Torch 400W',
                'corrosion': 'Sanding Disc 80G',
                'wear': 'Polishing Pad 3000G',
                'pitting': 'Filler Nozzle 0.5mm',
            }
            primary_type = max(set(defect_types), key=defect_types.count) if defect_types else 'unknown'
            tool_used = tool_map.get(primary_type, 'Multi-Tool Head')
            
            st.session_state.qa_report = {
                "job_id": str(uuid.uuid4())[:8].upper(),
                "cycle_time": cycle_time,
                "defects_total": num_defects,
                "defects_repaired": num_defects,
                "tool_used": tool_used,
                "status": "PASSED",
                "timestamp": st.session_state.get("execution_time", "Just now")
            }
        
        report = st.session_state.qa_report
        
        # Report Card Container with border
        with st.container(border=True):
            # Header row
            col_id, col_status = st.columns([3, 1])
            with col_id:
                st.markdown(f"**Job ID:** `{report['job_id']}`")
            with col_status:
                st.markdown(f"<span style='color: #2E7D32; font-weight: bold; font-size: 18px;'>{report['status']}</span>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cycle Time", f"{report['cycle_time']}s")
            m2.metric("Defects Repaired", f"{report['defects_repaired']}/{report['defects_total']}")
            m3.metric("Tool Used", report['tool_used'].split()[0])  # First word
            m4.metric("Quality Score", "98.5%", delta="+2.3%")
            
            # Tool detail caption
            st.caption(f"Primary Tool: **{report['tool_used']}**")
            
            st.markdown("---")
            
            # Download Report Button
            report_text = f"""
=====================================
     AARR QUALITY ASSURANCE REPORT
=====================================

Job ID:          {report['job_id']}
Status:          {report['status']}
Timestamp:       {report['timestamp']}

-------------------------------------
            REPAIR SUMMARY
-------------------------------------
Cycle Time:      {report['cycle_time']} seconds
Defects Found:   {report['defects_total']}
Defects Repaired:{report['defects_repaired']}
Success Rate:    100%

Tool Used:       {report['tool_used']}
Quality Score:   98.5%

-------------------------------------
          DEFECT DETAILS
-------------------------------------
"""
            for i, defect in enumerate(st.session_state.defects):
                report_text += f"""
Defect #{i+1}:
  Type:     {defect.get('type', 'Unknown').capitalize()}
  Severity: {defect.get('severity', 'Unknown').capitalize()}
  Position: {defect.get('position', (0,0,0))}
  Status:   REPAIRED
"""
            report_text += """
-------------------------------------
Certified by: AARR Autonomous System
=====================================
"""
            
            st.download_button(
                label="Download Report",
                data=report_text,
                file_name=f"AARR_Report_{report['job_id']}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Display simulation video if available
        if st.session_state.get("simulation_video"):
            video_path = st.session_state.simulation_video
            if Path(video_path).exists():
                st.markdown("##### Repair Execution Video")
                st.video(video_path)
            else:
                st.info("Video recording not available (PyBullet/OpenCV required)")

    
    # ============ ADVANCED TOOLS EXPANDER ============
    with st.expander("Advanced Tools", expanded=False):
        # ============ INTERACTIVE SEGMENTATION (SAM) ============
        st.markdown("##### Interactive Segmentation")
        
        sam_enabled = st.checkbox(
            "Enable Zero-Shot Segmentation",
            value=st.session_state.sam_enabled,
            key="sam_toggle",
            help="Click on the captured view to segment defects using SAM"
        )
        st.session_state.sam_enabled = sam_enabled
        
        if sam_enabled:
            if HAS_SAM:
                segmentor = get_segmentor()
                status = segmentor.get_status()
                if status["model_loaded"]:
                    st.caption("MobileSAM loaded")
                else:
                    st.caption("Using fallback (OpenCV)")
            else:
                st.caption("SAM not installed")
            
            # Capture snapshot button
            if st.button("Capture View", use_container_width=True, key="capture_sam"):
                if st.session_state.current_figure:
                    try:
                        img_bytes = st.session_state.current_figure.to_image(format="png", width=800, height=600)
                        st.session_state.sam_snapshot = img_bytes
                        st.session_state.sam_result = None
                    except Exception as e:
                        st.error(f"Capture failed: {e}")
            
            # Show click coordinate inputs when snapshot exists
            if st.session_state.sam_snapshot:
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.sam_click_x = st.number_input(
                        "Click X", min_value=0, max_value=800,
                        value=st.session_state.sam_click_x, step=10
                    )
                with col2:
                    st.session_state.sam_click_y = st.number_input(
                        "Click Y", min_value=0, max_value=600,
                        value=st.session_state.sam_click_y, step=10
                    )
                
                if st.button("SEGMENT", use_container_width=True, type="primary", key="segment_sam"):
                    if HAS_SAM:
                        import cv2
                        # Decode snapshot to numpy
                        nparr = np.frombuffer(st.session_state.sam_snapshot, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Run SAM segmentation
                        segmentor = get_segmentor()
                        result = segmentor.segment_at_point(
                            image,
                            st.session_state.sam_click_x,
                            st.session_state.sam_click_y
                        )
                        st.session_state.sam_result = result
                        st.rerun()
        
        st.markdown("---")
        
        # ============ SYNTHETIC DATA GENERATION ============
        st.markdown("##### Synthetic Data Pipeline")
        st.caption("Generate ML training datasets")
        
        # Only show if mesh is loaded
        can_generate = st.session_state.mesh_data is not None
        
        # Number of samples slider
        num_samples = st.slider(
            "Samples to generate",
            min_value=5,
            max_value=100,
            value=50,
            step=5,
            disabled=not can_generate,
            help="Number of training samples with randomized camera/lighting/defects"
        )
        
        if st.button("Generate Training Data", use_container_width=True, disabled=not can_generate, key="gen_data"):
            try:
                from src.simulation.synthetic_data_gen import generate_dataset
                
                output_dir = "synthetic_data"
                progress_bar = st.progress(0, text="Initializing...")
                status_text = st.empty()
                
                def update_progress(current, total):
                    progress = current / total
                    progress_bar.progress(progress, text=f"Generating sample {current}/{total}")
                    status_text.caption(f"Rendering scene with randomized camera/lighting...")
                
                # Generate dataset
                results = generate_dataset(
                    mesh_data=st.session_state.mesh_data,
                    num_samples=num_samples,
                    output_dir=output_dir,
                    progress_callback=update_progress
                )
                
                progress_bar.progress(1.0, text="Complete!")
                status_text.empty()
                st.success(f"Generated {len(results)} samples in `{output_dir}/`")
                st.caption(f"Images: `image_*.png` | Masks: `mask_*.png` | Meta: `metadata_*.json`")
                
            except Exception as e:
                st.error(f"Generation failed: {e}")
        
        if not can_generate:
            st.caption("Load a mesh first (STL/OBJ upload)")
    
    # ============ SYSTEM SETTINGS EXPANDER ============
    with st.expander("System Settings", expanded=False):
        st.markdown("##### Configuration")
        st.caption("System and workflow configuration options")
        
        # Reset button inside settings
        if step > 0:
            st.markdown("---")
            if st.button("↻ Reset Workflow", use_container_width=True, key="reset_workflow"):
                reset_state()
                st.session_state.chat_history = []
                st.rerun()
        
        # Additional system info
        st.markdown("---")
        st.caption(f"Connected: UR5e Robot")
        st.caption(f"Status: Online")
    
    # Standalone reset for quick access
    if step > 0:
        st.markdown("---")
        if st.button("↻ Reset", use_container_width=True, key="reset_main"):
            reset_state()
            st.session_state.chat_history = []
            st.rerun()


# ============ MAIN LAYOUT (70/30 Split) ============
# Note: Status info consolidated in viewport toolbar below


# Main 2-Column Layout
col_3d, col_chat = st.columns([0.7, 0.3])

# ============ LEFT COLUMN: 3D VIEWER ============
with col_3d:
    # === VIEWPORT CARD WITH HUD ===
    # Calculate vertex count for HUD
    vertex_count = "—"
    if st.session_state.mesh_data is not None:
        vertex_count = f"{len(st.session_state.mesh_data.vertices):,}"
    elif st.session_state.mesh_display_trace is not None:
        vertex_count = "Procedural"
    elif st.session_state.get('robot_traces') is not None:
        vertex_count = "Robot"
    
    st.markdown(f'''
    <div style="background: #F9FAFB; padding: 8px 12px; border-radius: 8px 8px 0 0; 
         display: flex; align-items: center; gap: 12px; font-family: 'Inter', -apple-system, sans-serif;
         border: 1px solid #E5E7EB; border-bottom: 2px solid #E5E7EB;">
        <span style="display: flex; align-items: center; gap: 6px;">
            <span style="width: 6px; height: 6px; background: #10B981; border-radius: 50%; 
                  animation: pulse-dot 2s infinite;"></span>
            <span style="color: #1F2937; font-weight: 600; font-size: 11px; text-transform: uppercase;">Online</span>
        </span>
        <span style="color: #D1D5DB;">|</span>
        <span style="display: flex; align-items: center; gap: 6px;">
            <span style="color: #6B7280; font-size: 11px;">ROBOT</span>
            <span style="color: #1F2937; font-weight: 600; font-size: 12px;">UR5e</span>
        </span>
        <span style="color: #D1D5DB;">|</span>
        <span style="display: flex; align-items: center; gap: 6px;">
            <span style="color: #6B7280; font-size: 11px;">PART</span>
            <span style="color: #1F2937; font-weight: 600; font-size: 12px;">{st.session_state.mesh_name}</span>
        </span>
        <span style="color: #D1D5DB;">|</span>
        <span style="display: flex; align-items: center; gap: 6px;">
            <span style="color: #6B7280; font-size: 11px;">DEFECTS</span>
            <span style="color: #DC2626; font-weight: 600; font-size: 12px;">{len(st.session_state.defects)}</span>
        </span>
    </div>
    ''', unsafe_allow_html=True)
    
    
    # Robot (Sci-Fi Drone) - special multi-trace rendering
    if st.session_state.get('robot_traces') is not None:
        fig = go.Figure(data=st.session_state.robot_traces)
        
        camera_eye = st.session_state.get('camera_eye', dict(x=1.5, y=1.5, z=1.0))
        
        # Sky blue background for concept art style
        sky_blue = 'rgb(33, 150, 243)'
        
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                bgcolor=sky_blue,
                xaxis=dict(visible=False, showbackground=False, showgrid=False),
                yaxis=dict(visible=False, showbackground=False, showgrid=False),
                zaxis=dict(visible=False, showbackground=False, showgrid=False),
                camera=dict(eye=camera_eye),
            ),
            paper_bgcolor=sky_blue,
            plot_bgcolor=sky_blue,
            height=480,
            margin=dict(l=0, r=0, b=0, t=0),
        )
        
        st.plotly_chart(fig, use_container_width=True, key="viewer_robot")
        st.session_state.current_figure = fig
    
    # Procedural mesh
    elif st.session_state.mesh_display_trace is not None:
        fig = go.Figure(data=[st.session_state.mesh_display_trace])
        
        # Read camera state from session (connects chatbot brain to viewer eyes)
        camera_eye = st.session_state.get('camera_eye', dict(x=1.5, y=1.5, z=1.5))
        
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                bgcolor='rgba(0,0,0,0)',  # ProAI: transparent for light UI
                xaxis=dict(visible=False, showbackground=False),
                yaxis=dict(visible=False, showbackground=False),
                zaxis=dict(visible=False, showbackground=False),
                camera=dict(eye=camera_eye),  # Apply camera from session state
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=480,
            margin=dict(l=0, r=0, b=0, t=0),
        )
        
        # Add custom path visualization if exists
        if st.session_state.custom_path_points:
            points = st.session_state.custom_path_points
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                z=[p[2] for p in points],
                mode='lines+markers',
                line=dict(color='#FF00FF', width=4),
                marker=dict(size=4, color='#FF00FF'),
                name=st.session_state.custom_path_info.get('pattern', 'Custom Path') if st.session_state.custom_path_info else 'Custom Path'
            ))
        
        st.plotly_chart(fig, use_container_width=True, key="viewer_3d")
        # Store figure for visual inspection
        st.session_state.current_figure = fig
    
    # Standard mesh
    elif st.session_state.mesh_data is not None:
        viewer = Mesh3DViewer(st.session_state.mesh_data)
        
        if st.session_state.defects:
            positions = [d['position'] for d in st.session_state.defects]
            labels = [d['type'].capitalize() for d in st.session_state.defects]
            severities = [d['severity'] for d in st.session_state.defects]
            viewer.add_defect_markers(positions, labels, severities, st.session_state.defect_normals)
        
        if st.session_state.highlight_position:
            viewer.highlight_region(st.session_state.highlight_position, radius=0.015)
        
        fig = viewer.create_figure()
        # ProAI Aesthetic: transparent background for light UI
        fig.update_layout(
            scene=dict(bgcolor='rgba(0,0,0,0)'),
            paper_bgcolor='rgba(0,0,0,0)',
            height=480,
        )
        
        if st.session_state.camera_target:
            camera = viewer.set_camera_view(st.session_state.camera_target)
            fig.update_layout(scene_camera=camera)
            # Also update camera_eye for consistency
            st.session_state.camera_eye = camera.get('eye', dict(x=1.5, y=1.5, z=1.5))
        else:
            # Default camera from session state
            camera_eye = st.session_state.get('camera_eye', dict(x=1.5, y=1.5, z=1.5))
            fig.update_layout(scene_camera=dict(eye=camera_eye))
        
        # Add custom path visualization if exists
        if st.session_state.custom_path_points:
            points = st.session_state.custom_path_points
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                z=[p[2] for p in points],
                mode='lines+markers',
                line=dict(color='#FF00FF', width=4),
                marker=dict(size=4, color='#FF00FF'),
                name=st.session_state.custom_path_info.get('pattern', 'Custom Path') if st.session_state.custom_path_info else 'Custom Path'
            ))
        
        st.plotly_chart(fig, use_container_width=True, key="viewer_mesh")
        # Store figure for visual inspection
        st.session_state.current_figure = fig
        
        # Note: Part stats now shown in HUD header above
    
    else:
        # ProAI: No part placeholder with light UI styling (bottom border-radius matches toolbar)
        st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-top: none; border-radius: 0 0 12px 12px; 
             height: 480px; display: flex; flex-direction: column; align-items: center; justify-content: center;
             box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
            <span style="font-size: 32px; margin-bottom: 16px; opacity: 0.3; font-weight: bold;">No Part</span>
            <span style="color: #6B6B6B; font-size: 18px; font-weight: 500;">No Part Loaded</span>
            <span style="color: #9A9A9A; font-size: 13px; margin-top: 8px;">Select a sample part or upload a 3D model</span>
        </div>
        """, unsafe_allow_html=True)
    
    # ============ SEGMENTATION RESULT DISPLAY ============
    if st.session_state.sam_enabled and (st.session_state.sam_snapshot or st.session_state.sam_result):
        st.markdown("---")
        st.markdown("##### Interactive Segmentation View")
        
        seg_col1, seg_col2 = st.columns(2)
        
        with seg_col1:
            st.markdown("**Captured Snapshot**")
            if st.session_state.sam_snapshot:
                st.image(
                    st.session_state.sam_snapshot,
                    caption=f"Click point: ({st.session_state.sam_click_x}, {st.session_state.sam_click_y})",
                    use_container_width=True
                )
        
        with seg_col2:
            st.markdown("**Segmentation Result**")
            if st.session_state.sam_result:
                result = st.session_state.sam_result
                st.image(
                    result.overlay,
                    caption=f"Coverage: {result.coverage_percent:.2f}% | Confidence: {result.confidence:.1%}",
                    use_container_width=True
                )
                
                # Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Coverage", f"{result.coverage_percent:.2f}%")
                col_m2.metric("Confidence", f"{result.confidence:.1%}")
                col_m3.metric("Click", f"({result.click_point[0]}, {result.click_point[1]})")
            else:
                st.info("Enter coordinates in sidebar and click SEGMENT")


# ============ RIGHT COLUMN: AI AGENT PANEL ============
with col_chat:
    st.markdown("""
    <div class="agent-header">
        <div class="agent-title">
            Factory Intelligence Unit
        </div>
        <div class="agent-subtitle">Supervisor · Inspector · Engineer</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container (compact height)
    chat_container = st.container(height=380)
    
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align: center; padding: 40px 20px; color: #9CA3AF;">
                <div style="font-size: 14px; font-weight: 500; margin-bottom: 8px;">Factory Assistant Ready</div>
                <div style="font-size: 12px;">Ask about defects, repairs, or inspections</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    with st.chat_message("user", avatar=None):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant", avatar=None):
                        st.write(msg["content"])
    
    # Input row: Voice button + Text input + Send button (side by side)
    col_voice, col_input, col_send = st.columns([0.12, 0.76, 0.12])
    
    with col_voice:
        with st.popover("Voice", help="Click to record voice command"):
            st.caption("Record your voice command:")
            audio_data = st.audio_input("Record", key="voice_input", label_visibility="collapsed")
            
            if audio_data:
                with st.spinner("Transcribing..."):
                    transcribed_text = transcribe_audio(audio_data.getvalue())
                
                if transcribed_text.startswith("Error:"):
                    st.error(transcribed_text)
                elif transcribed_text.strip():
                    st.success(f"Heard: {transcribed_text[:50]}...")
                    # Store in session state for processing
                    st.session_state["voice_transcript"] = transcribed_text
                    if st.button("Send Voice", key="send_voice_btn", type="primary", use_container_width=True):
                        process_user_message(transcribed_text)
                        st.session_state["voice_transcript"] = None
                        st.rerun()
                else:
                    st.warning("Could not transcribe audio. Please try again.")
    
    with col_input:
        user_text = st.text_input(
            "Message", 
            placeholder="Ask the team...", 
            key="chat_text_input",
            label_visibility="collapsed"
        )
    
    with col_send:
        send_clicked = st.button("Send", key="send_btn", type="primary", use_container_width=True)
    
    # Process text input when Send is clicked
    if send_clicked and user_text.strip():
        process_user_message(user_text)
        st.rerun()
    