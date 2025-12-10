# Changelog

All notable changes to the AARR project.

Format: [Semantic Versioning](https://semver.org/)

---

## [1.0.0] - 2024-12-10

### 🎉 Initial Release

Complete implementation of the Agentic Adaptive Repair Robot system.

### Added

#### Vision System
- HSV color-based defect detection (rust, crack, dent)
- **3D RGBD depth analysis** — Point cloud + curvature-based geometric detection
- SAM (Segment Anything) interactive segmentation
- Camera capture with RGB, depth, and segmentation
- Pixel-to-world coordinate transformation
- Surface normal estimation via ray casting

#### Agent System
- Multi-agent architecture (Supervisor, Inspector, Engineer)
- GPT-4o / Qwen3 LLM integration via LangGraph
- Function calling for UI control (zoom, scan, plan, execute)
- RAG knowledge base for SOP lookup
- Voice-to-text via OpenAI Whisper
- Code interpreter for custom path generation

#### Path Planning
- Spiral and raster toolpath patterns
- Surface normal alignment (tool perpendicular)
- TSP optimization for multi-defect ordering (NN + 2-opt)
- Custom path generation via LLM-written Python

#### Robot Control
- PyBullet simulation with KUKA iiwa 7-DOF arm
- Inverse kinematics solver
- Collision detection along path
- Workspace bounds validation

#### User Interface
- Streamlit web dashboard
- Interactive 3D Plotly viewer
- Premium industrial meshes (turbine, gear, pipe, gripper)
- Workflow: Scan → Plan → Approve → Execute
- Multi-agent chat panel

#### ML & Evaluation
- RandomForest repair time predictor
- Vision evaluation suite (precision/recall)
- Agent evaluation suite (tool accuracy)
- Synthetic data generation pipeline

#### Safety
- Human-in-the-loop approval
- Collision detection
- Singularity avoidance
- LLM fallback rules

### Documentation
- PRD.md — Product requirements
- ARCHITECTURE.md — System design
- MATH.md — Coordinate transforms
- SAFETY.md — Safety architecture
- PROMPTS.md — LLM prompting
- + 10 feature-specific docs

---

## [Unreleased]

### Added
- **Claude Aesthetic UI Theme** — Warm paper design inspired by Anthropic:
  - Background: Warm paper white (#FDFBF9)
  - Sidebar: Warm beige (#F4F1EA)
  - Headers: Merriweather serif font
  - Accent: Burnt orange (#D97757)
- **Texture-to-3D Defect Mapping** — `texture_analyzer.py` enables real scan-to-path workflow:
  - UV coordinate generation (planar, cylindrical, spherical)
  - Procedural rust texture generation
  - CV-based defect detection from textures
  - 2D texture → 3D vertex mapping
- **Light Mode 3D Viewer** — Adjusted lighting and colors for warm paper background:
  - Higher ambient (0.6) for contrast
  - Darker metal base (#A0A0A0)
  - Deep red defects (#C62828)
- **Pseudo-PBR Industrial Graphics** — High-end CAD/Three.js inspired rendering:
  - Specular: 1.5 (wet metal shine)
  - Fresnel: 0.5 (strong rim lighting)
  - Roughness: 0.1 (polished surface)
  - Reflection gradient for fake environment mapping
  - Industrial blue-grey background (#111827)
- **Studio Light PBR** — Optimized for Claude Light theme:
  - Gunmetal grey base (#707070) for light bg contrast
  - Specular: 1.8, Fresnel: 2.0 (studio quality shine)
  - Warm golden reflection gradient (sunset studio)
  - Burnt clay defects (#D97757) - sophisticated, not neon
  - Ground shadow plane for visual grounding
- **GLB Model Viewer** — Professional 3D rendering via Google model-viewer:
  - Supports local GLB/GLTF files
  - PBR materials and environment lighting
  - Auto-rotate and camera controls
  - Sample models included for testing
- **ProAI Aesthetic** — Premium industrial CAD render style:
  - Satin aluminum base (#F0F2F5) - light cool grey
  - High ambient (0.65), soft specular (0.4)
  - Fresnel 1.0 for rim separation on white bg
  - High-res geometry (100x100 grid = smooth curves)
  - Alert red defects (#DC2626) for contrast

### Planned
- [ ] Multi-view 3D reconstruction
- [ ] Real robot integration (ROS/RSI)
- [ ] Force feedback during execution
- [ ] Fine-tuned defect classification model
- [ ] Point cloud input from RealSense

---

## Version History

| Version | Date | Highlights |
|---------|------|-----------|
| 1.0.0 | 2024-12-10 | Full system with vision, agent, planning, control |
