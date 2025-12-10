# Product Requirements Document (PRD)
## Agentic Adaptive Repair Robot (AARR)

**Version**: 1.0  
**Last Updated**: December 2024  
**Author**: Amin Khalili

---

## 1. Executive Summary

### One-Liner
> An AI-powered robotic system that automatically detects surface defects and plans repair paths — with human-in-the-loop approval.

### The Problem
Manual inspection and repair of industrial parts is:
- **Slow** — Hours per part for skilled technicians
- **Expensive** — Requires expert operators
- **Inconsistent** — Human fatigue leads to missed defects
- **Unscalable** — High-mix/low-volume production is economically unviable

### The Solution
An **Agentic Scan-to-Path** system that:
1. **Scans** — Captures RGB + depth images of workpieces
2. **Detects** — Uses computer vision (2D color + 3D geometry) to find defects
3. **Decides** — LLM agent classifies severity and selects repair strategy
4. **Plans** — Generates optimized toolpaths with surface-perpendicular orientation
5. **Approves** — Human confirms before execution (safety checkpoint)
6. **Executes** — Robot follows path with collision avoidance

---

## 2. Target Users

### Primary Persona: Maintenance Engineer
- **Background**: Factory floor technician, not a programmer
- **Goal**: Repair defective parts quickly without writing robot code
- **Expectation**: Click "Scan" → Review results → Click "Approve" → Watch robot work

### Secondary Persona: AI/Robotics Interviewer
- **Background**: Technical evaluator at companies like Augmentus
- **Goal**: Assess full-stack AI + robotics integration skills
- **Expectation**: See real CV pipeline, LLM reasoning, IK control, safety layers

---

## 3. Functional Requirements

### FR-1: Defect Detection
| ID | Requirement | Status |
|----|-------------|--------|
| FR-1.1 | Detect rust via HSV color thresholding | ✅ Done |
| FR-1.2 | Detect cracks via dark region analysis | ✅ Done |
| FR-1.3 | Detect dents via 3D depth/curvature analysis | ✅ Done |
| FR-1.4 | Interactive SAM segmentation (click-to-mask) | ✅ Done |
| FR-1.5 | Multi-view 3D reconstruction | 🔜 Planned |

### FR-2: LLM Agent
| ID | Requirement | Status |
|----|-------------|--------|
| FR-2.1 | Natural language chat interface | ✅ Done |
| FR-2.2 | Function calling for UI control (zoom, scan, plan) | ✅ Done |
| FR-2.3 | RAG-based SOP lookup for repair parameters | ✅ Done |
| FR-2.4 | Multi-agent architecture (Supervisor/Inspector/Engineer) | ✅ Done |
| FR-2.5 | Voice-to-text commands via Whisper | ✅ Done |

### FR-3: Path Planning
| ID | Requirement | Status |
|----|-------------|--------|
| FR-3.1 | Spiral and raster toolpath patterns | ✅ Done |
| FR-3.2 | Surface normal alignment (tool perpendicular) | ✅ Done |
| FR-3.3 | TSP optimization for multi-defect ordering | ✅ Done |
| FR-3.4 | Custom path via code interpreter (LLM-generated Python) | ✅ Done |

### FR-4: Robot Execution
| ID | Requirement | Status |
|----|-------------|--------|
| FR-4.1 | PyBullet simulation with KUKA iiwa | ✅ Done |
| FR-4.2 | Inverse kinematics for path following | ✅ Done |
| FR-4.3 | Collision detection along path | ✅ Done |
| FR-4.4 | Real robot integration (ROS/RSI) | 🔜 Planned |

### FR-5: User Interface
| ID | Requirement | Status |
|----|-------------|--------|
| FR-5.1 | Streamlit web dashboard | ✅ Done |
| FR-5.2 | Interactive 3D Plotly viewer | ✅ Done |
| FR-5.3 | Workflow buttons: Scan → Plan → Approve → Execute | ✅ Done |
| FR-5.4 | Chat panel with multi-agent responses | ✅ Done |

---

## 4. Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| **Response Time** | Agent responds < 5 seconds |
| **Detection Accuracy** | > 90% recall on visible defects |
| **Local-First** | Run without internet (Ollama + Qwen) |
| **Safety** | Human approval required before execution |
| **Portability** | Works on Mac (M1/M2) and Linux |

---

## 5. Out of Scope (v1.0)

- ❌ Real production robot integration
- ❌ Force feedback during execution
- ❌ Multi-robot coordination
- ❌ Training custom ML models from factory data
- ❌ Real-time streaming from physical cameras

---

## 6. Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Defect Detection Rate** | > 90% | `eval_vision.py` on synthetic dataset |
| **Planning Time Saved** | > 80% vs manual | Time to generate path vs human programming |
| **User Workflow** | < 5 clicks | Scan → Detect → Plan → Approve → Execute |
| **Interview Demo** | < 10 min | Complete walkthrough for evaluator |

---

## 7. Technical Architecture

```
┌─────────────────────────────────────────┐
│           STREAMLIT DASHBOARD           │
│   [3D Viewer] [Chat] [Controls]         │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌────────┐  ┌─────────┐  ┌─────────┐
│ Vision │  │  Agent  │  │ Control │
│ OpenCV │  │LangGraph│  │PyBullet │
│ SAM    │  │ GPT-4o  │  │   IK    │
│ Depth  │  │  Qwen   │  │  Path   │
└────────┘  └─────────┘  └─────────┘
```

---

## 8. Key Differentiators

What makes this project stand out from "student demos":

| Aspect | Typical Demo | This Project |
|--------|--------------|--------------|
| **Vision** | Hardcoded positions | Real OpenCV + 3D depth analysis |
| **Robot** | Pre-programmed paths | IK with surface normal alignment |
| **AI** | Simple prompts | Multi-agent + RAG + function calling |
| **Safety** | None | Collision detection + human approval |
| **Polish** | CLI script | Full Streamlit dashboard |

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| PyBullet install fails (Mac) | Can't run simulation | Mock mode + Plotly-only demo |
| LLM hallucinates bad plans | Unsafe robot actions | Human approval + fallback rules |
| No real camera data | Can't prove real-world value | Synthetic data pipeline |
| Interview time pressure | Can't show all features | Prepared 5-min demo script |

---

## 10. Demo Script (5 minutes)

1. **Open** — `streamlit run app/streamlit_app.py`
2. **Show** — 3D turbine blade with defect markers
3. **Chat** — "Show me the worst defect" → Camera zooms
4. **Voice** — Click mic, say "Plan the repair" → Path appears
5. **Explain** — Point out spiral path + surface normals
6. **Execute** — Click Approve → Watch simulation
7. **Q&A** — "How does the 3D vision work?" → Explain RGBD pipeline
