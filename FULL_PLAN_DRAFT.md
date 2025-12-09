# Agentic Adaptive Repair Robot (AARR)
## Complete Implementation Plan (Opus 4.5 + Gemini-3 Collaboration)

**Author:** [Your Name]  
**Target Role:** AI Engineer @ Augmentus  
**Estimated Development Time:** ~17 hours  
**Last Updated:** December 9, 2024

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Technical Architecture](#3-technical-architecture)
4. [Project Structure](#4-project-structure)
5. [Core Features](#5-core-features)
6. [Advanced Features](#6-advanced-features)
7. [Documentation Suite](#7-documentation-suite)
8. [Implementation Phases](#8-implementation-phases)
9. [Technical Deep Dives](#9-technical-deep-dives)
10. [Verification Plan](#10-verification-plan)
11. [Demo Strategy](#11-demo-strategy)

---

## 1. Executive Summary

### What This Project Is
A simulated industrial robot system that **automatically detects surface defects** (rust, cracks, dents) on workpieces and **autonomously plans and executes repairs** using an LLM-based decision agent. This demonstrates the "Scan-to-Path" workflow that Augmentus sells commercially.

### Why This Project Matters for Augmentus
- **Computer Vision:** Real OpenCV detection pipeline (not mocked)
- **Robotics Control:** Inverse Kinematics with surface normal alignment
- **AI/LLM Integration:** LangGraph agent with tool-calling, memory, and fallback logic
- **Full-Stack:** Streamlit dashboard for operator interface
- **Production Thinking:** Docker, CI/CD, observability (LangSmith), safety constraints

### Key Differentiators from "Student Projects"
1. **Real perception** — not `p.getBasePositionAndOrientation()` cheating
2. **Surface normal alignment** — tool stays perpendicular to curved surfaces
3. **Industrial documentation** — MATH.md, PROMPTS.md, SAFETY.md
4. **Human-in-the-loop** — approval step before execution
5. **Graceful degradation** — fallback when LLM fails

---

## 2. Problem Statement

### Context
In MRO (Maintenance, Repair, and Overhaul) industries—aerospace, marine, automotive—parts arrive with **unpredictable damage** (rust, cracks, dents) in **varying locations**.

### The Pain Point
Traditional industrial robots require operators to **manually jog the robot** to teach waypoints for every unique defect. This is:
- Time-consuming (hours per part)
- Economically unviable for "High-Mix / Low-Volume" production
- Requires skilled operators

### The Solution
An **Agentic Scan-to-Path** system that:
1. **Scans** — Captures image, detects defects automatically
2. **Decides** — LLM agent classifies severity and selects repair strategy
3. **Plans** — Generates toolpath (spiral/raster) with proper tool orientation
4. **Executes** — Robot follows path with collision avoidance
5. **Verifies** — Re-scans to confirm repair success

### User Persona
**Maintenance Engineer (Non-programmer)**
- Goal: Repair a batch of rusty pipes without writing code or manually guiding the robot
- Expectation: Click "Scan" → Review → Click "Approve" → Watch robot work

---

## 3. Technical Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STREAMLIT DASHBOARD                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Camera Feed │  │ Defect List │  │ Path Preview│  │ Performance Metrics │ │
│  │   + Scan    │  │ + Confidence│  │    (3D)     │  │  Scan/Plan/Execute  │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────────────────────┘ │
│         │                │                │                                  │
│  [Dark Mode] [Keyboard Shortcuts: S=Scan, Enter=Approve, R=Reset]           │
└─────────┼────────────────┼────────────────┼─────────────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LANGGRAPH AGENT (GPT-4)                              │
│                                                                              │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│   │  DETECT  │──▶│ CLASSIFY │──▶│  MEMORY  │──▶│   PLAN   │──▶│ EXECUTE  │  │
│   │ defects  │   │ severity │   │  lookup  │   │  path    │   │  path    │  │
│   └──────────┘   └──────────┘   └──────────┘   └────┬─────┘   └──────────┘  │
│                                                     │                        │
│                                              [HUMAN APPROVAL]                │
│                                                     │                        │
│   ┌──────────────────────────────────────┐         │                        │
│   │ FALLBACK (if LLM fails 3x)           │◀────────┘                        │
│   │ Rule-based: rust→spiral, crack→raster│                                  │
│   └──────────────────────────────────────┘                                  │
│                                                                              │
│   [LangSmith Tracing Enabled]                                               │
└──────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          COMPUTER VISION MODULE                              │
│                                                                              │
│   Camera Capture (RGB + Depth)                                              │
│         │                                                                    │
│         ▼                                                                    │
│   HSV Color Thresholding                                                    │
│   - Rust: Red (H: 0-10, 170-180)                                            │
│   - Crack: Black (V < 50)                                                   │
│   - Dent: Blue (H: 100-130)                                                 │
│         │                                                                    │
│         ▼                                                                    │
│   Contour Detection + Bounding Boxes                                        │
│         │                                                                    │
│         ▼                                                                    │
│   Confidence Scoring (area-based)                                           │
│         │                                                                    │
│         ▼                                                                    │
│   Pixel → 3D World Coordinates                                              │
│   P_world = T_cam_to_base · K^(-1) · [u, v, 1]^T · depth                   │
│         │                                                                    │
│         ▼                                                                    │
│   ★ SURFACE NORMAL ESTIMATION ★                                             │
│   - Ray cast from camera to defect                                          │
│   - Get hit_normal from PyBullet                                            │
│   - Calculate tool orientation quaternion                                   │
└──────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PATH PLANNING MODULE                               │
│                                                                              │
│   Multi-Defect TSP Ordering                                                 │
│   - Nearest-neighbor algorithm                                              │
│   - Minimize total travel distance                                          │
│         │                                                                    │
│         ▼                                                                    │
│   Material-Specific Strategy Selection                                      │
│   - Steel: sanding_disc_80, spiral pattern                                  │
│   - Aluminum: polishing_pad, gentle_spiral                                  │
│   - Composite: vacuum_sander, low_pressure_raster                           │
│         │                                                                    │
│         ▼                                                                    │
│   Toolpath Generation                                                       │
│   - Spiral: Archimedean spiral from center outward                          │
│   - Raster: Back-and-forth lines with overlap                               │
│         │                                                                    │
│         ▼                                                                    │
│   Cost/Time Estimation                                                      │
│   - Based on path length and defect area                                    │
└──────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ROBOT CONTROL MODULE                                 │
│                                                                              │
│   ★ INVERSE KINEMATICS WITH ORIENTATION ★                                   │
│   - Position target: [x, y, z]                                              │
│   - Orientation target: quaternion [qx, qy, qz, qw]                         │
│   - Tool Z-axis aligned with surface normal                                 │
│         │                                                                    │
│         ▼                                                                    │
│   Safety Checks (Before Each Move)                                          │
│   - Collision detection (PyBullet getClosestPoints)                         │
│   - Singularity avoidance (Jacobian condition number)                       │
│   - Workspace bounds validation                                             │
│         │                                                                    │
│         ▼                                                                    │
│   Motor Control                                                             │
│   - p.setJointMotorControl2 for each joint                                  │
│         │                                                                    │
│         ▼                                                                    │
│   Visual Feedback                                                           │
│   - Defect turns grey when robot touches it                                 │
│   - p.changeVisualShape(defect_id, rgbaColor=[0.5, 0.5, 0.5, 1])           │
└──────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PYBULLET SIMULATION                                │
│                                                                              │
│   World Setup                                                               │
│   - Ground plane                                                            │
│   - KUKA iiwa robot (7-DOF, orange, industrial)                             │
│   - Workpiece (cube/cylinder/custom .obj)                                   │
│   - Overhead camera                                                         │
│                                                                              │
│   Defect Markers                                                            │
│   - Rust: Red sphere (radius 0.03m)                                         │
│   - Crack: Black line                                                       │
│   - Dent: Blue concave marker                                               │
│                                                                              │
│   Visualization                                                             │
│   - 3D path lines before execution                                          │
│   - Safety zone boxes (red transparent)                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Project Structure

```
augmentus-repair-robot/
│
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions: pytest + flake8
│
├── .env.example                  # API key templates (OPENAI, LANGCHAIN)
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md                     # Executive summary + demo GIF + architecture
├── CHANGELOG.md                  # Version history
│
├── docs/                         # 6 PROFESSIONAL DOCUMENTS
│   ├── PRD.md                    # Product Requirements Document
│   ├── ARCHITECTURE.md           # Code structure and data flow
│   ├── MATH.md                   # Coordinate frames + linear algebra
│   ├── PROMPTS.md                # Agent system card + tool definitions
│   ├── SAFETY.md                 # Industrial constraints
│   └── SIM_TO_REAL.md            # Calibration + deployment strategy
│
├── src/
│   ├── __init__.py
│   ├── main.py                   # CLI entry point
│   ├── config.py                 # YAML config loader
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── environment.py        # PyBullet world + KUKA robot
│   │   ├── defects.py            # Spawn rust/crack/dent markers
│   │   └── visualizer.py         # 3D path lines, safety zones
│   │
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── camera.py             # Capture RGB + Depth
│   │   ├── detector.py           # HSV threshold + contours + confidence
│   │   ├── localization.py       # Pixel → 3D coordinates
│   │   └── normal_estimator.py   # Ray cast + surface normal
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py              # LangGraph StateGraph workflow
│   │   ├── tools.py              # @tool decorated functions
│   │   ├── prompts.py            # System prompt for Process Engineer
│   │   ├── cost_estimator.py     # Time/cost prediction
│   │   ├── memory.py             # Session memory lookup
│   │   └── fallback.py           # Rule-based fallback when LLM fails
│   │
│   ├── planning/
│   │   ├── __init__.py
│   │   ├── paths.py              # Spiral + raster generators
│   │   └── tsp.py                # Multi-defect ordering
│   │
│   ├── control/
│   │   ├── __init__.py
│   │   └── controller.py         # IK solver + motor control
│   │
│   ├── safety/
│   │   ├── __init__.py
│   │   ├── collision.py          # Collision detection
│   │   └── singularity.py        # Jacobian condition check
│   │
│   ├── data/
│   │   └── materials.yaml        # Material-specific strategies
│   │
│   └── reports/
│       ├── __init__.py
│       ├── exporter.py           # JSON report generation
│       └── history.py            # Session logging
│
├── app/
│   └── streamlit_app.py          # Full dashboard UI
│
├── tests/
│   ├── test_vision.py
│   ├── test_planning.py
│   ├── test_agent.py
│   └── test_safety.py
│
└── assets/
    ├── scenes/
    │   ├── demo_cube.yaml        # Default demo scene
    │   ├── demo_pipe.yaml        # Cylindrical workpiece
    │   └── demo_complex.yaml     # Stress test (8+ defects)
    └── demo.gif                  # Screen recording
```

---

## 5. Core Features

### 5.1 Real Computer Vision (Not Mocked)

| Feature | Implementation |
|---------|----------------|
| **Camera Capture** | `p.getCameraImage()` returns RGB + Depth |
| **Color Detection** | HSV thresholding per defect type |
| **Multi-Defect** | `cv2.findContours()` detects all defects |
| **Confidence %** | Based on contour area and color match quality |
| **3D Localization** | Pixel → Camera → World coordinate transform |

### 5.2 Surface Normal Alignment

```python
def get_orientation_from_normal(normal_vector):
    """
    Calculate quaternion to align tool Z-axis with surface normal.
    """
    target = -np.array(normal_vector)  # Point INTO surface
    source = np.array([0, 0, 1])       # Default tool axis
    
    axis = np.cross(source, target)
    angle = np.arccos(np.dot(source, target))
    
    rot = Rotation.from_rotvec(axis * angle)
    return rot.as_quat()
```

**Why this matters:** Without orientation, the sanding disc edge digs into curved surfaces. This math is the core of Augmentus's Scan-to-Path engine.

### 5.3 LangGraph Agent

```
State Machine:
detect → classify → check_memory → estimate_cost → plan → [HUMAN APPROVAL] → execute → verify
```

**Tools available to agent:**
- `detect_defects()` — Trigger vision pipeline
- `classify_severity(defect)` — minor/moderate/severe
- `get_material_properties(material)` — Lookup repair strategy
- `plan_spiral_path(center, radius)` — Generate toolpath
- `estimate_repair_time(area)` — Predict duration
- `execute_path(waypoints)` — Send to robot
- `verify_repair()` — Re-scan after completion

### 5.4 KUKA iiwa Robot Control

- **Robot:** KUKA iiwa (7-DOF, industrial, built into PyBullet)
- **IK Solver:** `p.calculateInverseKinematics()` with position + orientation
- **Motor Control:** `p.setJointMotorControl2()` for smooth motion

### 5.5 Streamlit Dashboard

```
┌────────────────────────────────────────────────────────────────┐
│  🔧 AARR Control Panel                    [Scene ▼] [🌙 Dark]  │
├──────────────────┬─────────────────────────────────────────────┤
│  📷 Camera Feed  │  📊 Metrics: Scan 0.8s | Plan 1.2s | Exec 12s │
│  [Live Image]    │                                              │
│                  │  🎯 Detected Defects                        │
│  [🔍 Scan] [S]   │  #1 RUST  (0.5, 0.1, 0.4) 92% ██████████   │
│                  │  #2 CRACK (0.6, -0.1, 0.4) 78% ████████░░   │
│  Path Preview:   │  #3 DENT  (0.55, 0.0, 0.4) 85% █████████░   │
│  [3D Matplotlib] │                                              │
│                  │  💰 Estimate: 45s total                     │
│  [✅ Approve]    │                                              │
│  [❌ Reject]     │  📜 Agent Log                               │
│  [🔄 Reset] [R]  │  > Scanning workspace...                    │
│                  │  > 3 defects found                          │
│                  │  > Planning spiral for RUST...              │
├──────────────────┴─────────────────────────────────────────────┤
│  📁 History  │  📄 Export Report                               │
└────────────────────────────────────────────────────────────────┘
```

---

## 6. Advanced Features

### 6.1 Multi-Defect + TSP Ordering
```python
def optimize_order(defects):
    """Nearest-neighbor TSP to minimize travel."""
    ordered = [defects[0]]
    remaining = defects[1:]
    while remaining:
        last = ordered[-1]
        nearest = min(remaining, key=lambda d: distance(last.pos, d.pos))
        ordered.append(nearest)
        remaining.remove(nearest)
    return ordered
```

### 6.2 Material Database
```yaml
# data/materials.yaml
steel:
  hardness: high
  tools: [sanding_disc_80, sanding_disc_120]
  strategies: [spiral, raster]
  
aluminum:
  hardness: medium
  tools: [polishing_pad]
  strategies: [gentle_spiral]
  
composite:
  hardness: low
  tools: [vacuum_sander]
  strategies: [low_pressure_raster]
```

### 6.3 Graceful Degradation
```python
def plan_with_fallback(defect, max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm_plan(defect)  # Try LLM
        except LLMError:
            continue
    
    # Fallback to rule-based
    if defect.type == "rust":
        return {"strategy": "spiral", "tool": "sanding_disc_80"}
    elif defect.type == "crack":
        return {"strategy": "raster", "tool": "filler_applicator"}
```

### 6.4 Agent Memory
```python
def check_memory(defect):
    similar = memory.search(
        query=f"{defect.type} on {defect.material}",
        limit=3
    )
    if similar:
        return f"Similar defect fixed with {similar[0].strategy}, success rate: 95%"
    return None
```

### 6.5 Human-in-the-Loop
- After `plan` step, workflow **pauses**
- Streamlit shows path preview + cost estimate
- User must click **Approve** or **Reject**
- Audit log records all approvals with timestamp

### 6.6 Edge Case Handling

| Case | Behavior |
|------|----------|
| No defects | "Workspace clear" message |
| >10 defects | Batch with progress bar |
| Unreachable | Skip + log warning |
| Low confidence (<50%) | Flag for manual review |
| LLM timeout | Fallback to rules |

### 6.7 Safety Layer

**Collision Detection:**
```python
def check_collision(robot_id, obstacles):
    for obs in obstacles:
        contacts = p.getClosestPoints(robot_id, obs, distance=0.01)
        if contacts:
            return True
    return False
```

**Singularity Avoidance:**
```python
def near_singularity(joint_positions):
    J = compute_jacobian(robot_id, joint_positions)
    return np.linalg.cond(J) > 100  # Threshold
```

### 6.8 Visual Feedback
```python
# Defect "disappears" when repaired
def mark_repaired(defect_id):
    p.changeVisualShape(defect_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1])
```

### 6.9 LangSmith Observability
```python
# .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=xxx
LANGCHAIN_PROJECT=aarr

# Full trace of: detect → classify → plan → execute → verify
```

---

## 7. Documentation Suite (6 Documents)

### 7.1 PRD.md — Product Requirements
- Executive Summary
- Problem Statement (MRO pain point)
- User Personas
- Functional Requirements (FR 1-5)
- Non-Functional Requirements
- Success Metrics (KPIs)

### 7.2 ARCHITECTURE.md — Code Structure
- Module diagram
- Data flow
- API contracts between modules
- Dependency graph

### 7.3 MATH.md — Coordinate Frame Atlas

**Purpose:** Answer the interview question before they ask it.

**Contents:**
1. **Coordinate Frame Diagram**
   - Camera (C) → World/Base (W) → Tool/TCP (T) → Object (O)

2. **Pixel-to-World Pipeline**
   ```
   P_camera = K^(-1) · [u, v, 1]^T
   P_world = T_base_to_camera · P_camera · depth
   ```

3. **Surface Normal → Tool Orientation**
   ```
   axis = cross([0,0,1], normal)
   angle = arccos(dot([0,0,1], normal))
   quaternion = axis_angle_to_quat(axis, angle)
   ```

### 7.4 PROMPTS.md — Agent System Card

**Purpose:** Show LLM-as-a-software-component thinking.

**Contents:**
1. **System Prompt** (full text)
2. **Tool Definitions** (all @tool functions)
3. **Failure Handling** (validation, retry, fallback)
4. **Output Schema** (JSON format)

### 7.5 SAFETY.md — Industrial Constraints

**Purpose:** Show safety-critical thinking.

**Contents:**
1. **Collision Avoidance** — PyBullet `getClosestPoints`
2. **Singularity Handling** — Jacobian condition number
3. **Workspace Bounds** — Validate before execution
4. **Human-in-the-Loop** — Approval requirement rationale

### 7.6 SIM_TO_REAL.md — Deployment Strategy

**Key Statement:**
> "In this simulation, the camera and robot share a perfect coordinate system. In a real-world deployment, I would implement Hand-Eye Calibration (e.g., Tsai-Lenz algorithm) to map the camera frame to the robot base frame."

**Contents:**
1. Hand-Eye Calibration
2. TCP Calibration
3. Workspace Registration
4. Safety System Integration (E-stops, light curtains)

---

## 8. Implementation Phases

| # | Phase | Time | Key Deliverables |
|---|-------|------|------------------|
| 1 | Scaffolding | 30 min | `.env`, `requirements.txt`, `Dockerfile`, CI/CD |
| 2 | Simulation | 1 hr | PyBullet world, KUKA robot, defect markers |
| 3 | Vision | 2 hr | Camera, HSV detection, confidence scoring |
| 4 | Surface Normal | 45 min | Ray casting, orientation quaternion |
| 5 | Material DB | 30 min | `materials.yaml`, strategy lookup |
| 6 | Path Planning | 1.5 hr | Spiral/raster generators, TSP |
| 7 | Robot Control | 1.5 hr | IK with orientation, motor control |
| 8 | Safety Layer | 45 min | Collision, singularity, bounds |
| 9 | LangGraph Agent | 2.5 hr | Full workflow, tools, LangSmith |
| 10 | Agent Memory | 30 min | Session storage, similarity search |
| 11 | Human-in-the-Loop | 30 min | Approval pause, audit log |
| 12 | Edge Cases | 30 min | Empty, overflow, unreachable |
| 13 | Reports | 45 min | JSON export, session history |
| 14 | Streamlit UI | 2 hr | Full dashboard with all features |
| 15 | Documentation | 2 hr | All 6 docs |
| 16 | Demo & Polish | 1 hr | Split-screen recording, GIF |
| **TOTAL** | | **~17 hr** | |

---

## 9. Technical Deep Dives

### 9.1 The Pixel-to-World Transform

```python
def pixel_to_world(u, v, depth, K, T_cam_to_base):
    """
    Convert pixel coordinates to world coordinates.
    
    Args:
        u, v: Pixel coordinates
        depth: Depth value at (u, v)
        K: 3x3 camera intrinsic matrix
        T_cam_to_base: 4x4 camera-to-base transform
    
    Returns:
        [x, y, z] in world frame
    """
    # Step 1: Pixel to normalized camera coordinates
    K_inv = np.linalg.inv(K)
    p_norm = K_inv @ np.array([u, v, 1])
    
    # Step 2: Scale by depth
    p_camera = p_norm * depth
    
    # Step 3: Transform to world frame
    p_camera_homo = np.append(p_camera, 1)  # Homogeneous
    p_world_homo = T_cam_to_base @ p_camera_homo
    
    return p_world_homo[:3]
```

### 9.2 The Surface Normal Calculation

```python
def get_surface_normal(camera_pos, target_pos):
    """
    Ray cast from camera to target, get surface normal.
    """
    result = p.rayTest(camera_pos, target_pos)
    
    if result[0][0] != -1:  # Hit something
        hit_position = result[0][3]
        hit_normal = result[0][4]  # [nx, ny, nz]
        return hit_normal
    
    return [0, 0, 1]  # Default up

def normal_to_quaternion(normal):
    """
    Calculate quaternion to align tool Z-axis with normal.
    """
    target = -np.array(normal)  # Point INTO surface
    target = target / np.linalg.norm(target)
    
    source = np.array([0, 0, 1])
    
    # Handle edge cases
    dot = np.dot(source, target)
    if dot > 0.9999:
        return [0, 0, 0, 1]  # Identity
    if dot < -0.9999:
        return [1, 0, 0, 0]  # 180° flip
    
    axis = np.cross(source, target)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    
    rot = Rotation.from_rotvec(axis * angle)
    return rot.as_quat()
```

### 9.3 The LangGraph Workflow

```python
from langgraph.graph import StateGraph, END

class RepairState(TypedDict):
    defects: List[Defect]
    current_defect: Defect
    path: List[Pose]
    status: str
    approved: bool

def create_workflow():
    graph = StateGraph(RepairState)
    
    # Nodes
    graph.add_node("detect", detect_node)
    graph.add_node("classify", classify_node)
    graph.add_node("memory", memory_node)
    graph.add_node("plan", plan_node)
    graph.add_node("await_approval", approval_node)
    graph.add_node("execute", execute_node)
    graph.add_node("verify", verify_node)
    
    # Edges
    graph.add_edge("detect", "classify")
    graph.add_edge("classify", "memory")
    graph.add_edge("memory", "plan")
    graph.add_edge("plan", "await_approval")
    graph.add_conditional_edges(
        "await_approval",
        lambda s: "execute" if s["approved"] else "plan"
    )
    graph.add_edge("execute", "verify")
    graph.add_conditional_edges(
        "verify",
        lambda s: "classify" if s["defects"] else END
    )
    
    return graph.compile()
```

---

## 10. Verification Plan

### Automated Tests (CI/CD)

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
      - run: flake8 src/ --max-line-length=120
```

### Manual Verification Checklist

- [ ] PyBullet window shows KUKA + workpiece + defects
- [ ] Camera captures correct view
- [ ] Vision detects all 3 defect types
- [ ] Confidence % displayed correctly
- [ ] Surface normals point into surface
- [ ] Robot tool stays perpendicular during path
- [ ] TSP orders defects efficiently
- [ ] Material strategies applied correctly
- [ ] Agent memory recalls past repairs
- [ ] Graceful fallback when LLM fails
- [ ] Human approval blocks execution
- [ ] Edge cases handled (empty, overflow)
- [ ] Collision detection stops dangerous moves
- [ ] Defects turn grey after repair
- [ ] LangSmith trace shows full workflow
- [ ] JSON report exports correctly
- [ ] Keyboard shortcuts work
- [ ] Dark mode toggles
- [ ] All 6 docs complete and accurate
- [ ] Demo GIF plays in README

---

## 11. Demo Strategy

### Split-Screen Recording

| Left Side | Right Side |
|-----------|------------|
| Streamlit UI | PyBullet 3D Window |
| Camera feed | Robot in world |
| Agent log scrolling | Robot moving |
| "APPROVED" click | Defect turning grey |

### Key Moments to Capture

1. **Scan:** Defects appear with confidence %
2. **Agent Reasoning:** Log shows "Rust detected... planning spiral..."
3. **Approval:** User clicks ✅ Approve
4. **Execution:** Robot follows spiral path (tool perpendicular!)
5. **Repair:** Rust turns grey on contact
6. **Verify:** "Repair complete, 0 defects remaining"

### Technical Requirements

- Duration: 60-90 seconds
- Format: MP4 → convert to GIF for README
- Resolution: 1920x1080 (both windows visible)
- Tool: OBS Studio or QuickTime

### README Embed

```markdown
## Demo

![AARR Demo](assets/demo.gif)

*Split-screen showing Streamlit UI (left) and PyBullet simulation (right). 
The robot detects rust, plans a spiral path, awaits approval, then executes 
the repair while maintaining tool perpendicularity to the surface.*
```

---

## Summary: Why This Project Wins

| Aspect | What We Built | Why Augmentus Cares |
|--------|--------------|---------------------|
| **Vision** | Real OpenCV, not mocked | They build real perception systems |
| **Orientation** | Surface normal alignment | Core of their Scan-to-Path tech |
| **Agent** | LangGraph + GPT-4 + tools | They're hiring for Agentic AI |
| **Safety** | Collision, singularity, HITL | Industrial robots are dangerous |
| **Full-Stack** | Streamlit dashboard | They need engineers who can demo |
| **Docs** | MATH.md, PROMPTS.md, SAFETY.md | Shows engineering maturity |
| **Production** | Docker, CI, LangSmith | Shows deployment readiness |

---

## Questions for Codex Review

1. Is the architecture sound? Any missing components?
2. Are there better approaches for any of the technical deep-dives?
3. Is the documentation strategy appropriate for the role?
4. Any suggestions to make this even more impressive?
5. Estimated time realistic (~17 hours)?

---

*This plan is the combined work of Opus 4.5 and Gemini-3, refined through iterative feedback.*
