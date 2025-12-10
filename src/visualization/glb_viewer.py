"""
GLB Model Viewer Component for Streamlit.

Uses Google's <model-viewer> web component for high-quality GLB/GLTF rendering.
This provides Three.js-level quality without heavy Python dependencies.

Usage:
    from src.visualization.glb_viewer import render_glb_viewer
    
    render_glb_viewer("assets/models/part.glb", height=500)
"""

import streamlit as st
import streamlit.components.v1 as components
import base64
from pathlib import Path
from typing import Optional


def render_glb_viewer(
    glb_path: str,
    height: int = 500,
    auto_rotate: bool = True,
    camera_controls: bool = True,
    background_color: str = "transparent",
    exposure: float = 1.0,
    shadow_intensity: float = 1.0,
    environment_image: Optional[str] = None,
) -> None:
    """
    Render a GLB/GLTF 3D model using Google's model-viewer.
    
    This provides professional-grade 3D rendering with:
    - PBR materials
    - Environment-based lighting
    - Smooth camera controls
    - AR support (on mobile)
    
    Args:
        glb_path: Path to GLB/GLTF file
        height: Viewer height in pixels
        auto_rotate: Enable auto-rotation
        camera_controls: Enable mouse camera controls
        background_color: CSS color or "transparent"
        exposure: Light exposure (0.5-2.0)
        shadow_intensity: Ground shadow intensity (0-2)
        environment_image: Optional HDR/skybox URL
    """
    # Read and encode GLB file
    path = Path(glb_path)
    if not path.exists():
        st.error(f"GLB file not found: {glb_path}")
        return
    
    with open(path, "rb") as f:
        glb_data = base64.b64encode(f.read()).decode("utf-8")
    
    # Build model-viewer attributes
    attrs = []
    if auto_rotate:
        attrs.append('auto-rotate')
    if camera_controls:
        attrs.append('camera-controls')
    
    attrs_str = " ".join(attrs)
    
    # Environment/lighting setup
    env_attr = ""
    if environment_image:
        env_attr = f'environment-image="{environment_image}"'
    else:
        # Use neutral studio lighting
        env_attr = 'environment-image="neutral"'
    
    # HTML with model-viewer web component
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.3.0/model-viewer.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: {background_color};
            }}
            model-viewer {{
                width: 100%;
                height: {height}px;
                background-color: {background_color};
                --poster-color: transparent;
            }}
            model-viewer::part(default-progress-bar) {{
                display: none;
            }}
        </style>
    </head>
    <body>
        <model-viewer
            src="data:model/gltf-binary;base64,{glb_data}"
            {attrs_str}
            {env_attr}
            exposure="{exposure}"
            shadow-intensity="{shadow_intensity}"
            shadow-softness="0.5"
            tone-mapping="commerce"
            interaction-prompt="none"
        >
        </model-viewer>
    </body>
    </html>
    """
    
    components.html(html_content, height=height + 20)


def render_glb_from_url(
    glb_url: str,
    height: int = 500,
    auto_rotate: bool = True,
    camera_controls: bool = True,
    background_color: str = "transparent",
) -> None:
    """
    Render a GLB model from a URL (e.g., public CDN or Sketchfab).
    
    Args:
        glb_url: Public URL to GLB file
        height: Viewer height in pixels
        auto_rotate: Enable auto-rotation
        camera_controls: Enable mouse camera controls
        background_color: CSS background color
    """
    attrs = []
    if auto_rotate:
        attrs.append('auto-rotate')
    if camera_controls:
        attrs.append('camera-controls')
    
    attrs_str = " ".join(attrs)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.3.0/model-viewer.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: {background_color};
            }}
            model-viewer {{
                width: 100%;
                height: {height}px;
                background-color: {background_color};
            }}
        </style>
    </head>
    <body>
        <model-viewer
            src="{glb_url}"
            {attrs_str}
            environment-image="neutral"
            exposure="1"
            shadow-intensity="1"
            tone-mapping="commerce"
        >
        </model-viewer>
    </body>
    </html>
    """
    
    components.html(html_content, height=height + 20)


# Sample GLB URLs for testing (public domain / CC0)
SAMPLE_GLBS = {
    "astronaut": "https://modelviewer.dev/shared-assets/models/Astronaut.glb",
    "robot": "https://modelviewer.dev/shared-assets/models/RobotExpressive.glb",
    "damaged_helmet": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/DamagedHelmet/glTF-Binary/DamagedHelmet.glb",
}
