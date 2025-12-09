# Running Tests

All commands should be run from the project root:

```bash
cd "/Users/amin/dev/Robotic AI"
```

---

## Quick Reference

| Test | Command | Time |
|------|---------|------|
| Basic simulation | `python -m src.main --demo` | 2s |
| Vision only | `python -m src.test_vision` | 5s |
| Full pipeline | `python -m src.test_pipeline` | 15s |
| Agent (Qwen3) | `python -m src.test_agent` | 60-180s |
| Dashboard | `streamlit run app/streamlit_app.py` | Interactive |

**Note:** Use `/opt/anaconda3/bin/python` if the default Python doesn't have PyBullet.

---

## Test 1: Basic Simulation

**Purpose:** Verify PyBullet loads correctly with robot and defects

```bash
/opt/anaconda3/bin/python -m src.main --demo
```

**Expected:**
- PyBullet window opens
- KUKA iiwa robot visible
- Grey workpiece with 3 colored defect markers
- Press `Ctrl+C` to exit

---

## Test 2: Vision Pipeline

**Purpose:** Test camera capture, HSV detection, and 3D localization

```bash
/opt/anaconda3/bin/python -m src.test_vision
```

**Expected Output:**
```
[1/5] Setting up simulation...
  ✓ Created 3 defects
[2/5] Initializing vision pipeline...
  ✓ Camera, Detector, Localizer ready
[3/5] Capturing image...
  ✓ Captured RGB: (480, 640, 3)
[4/5] Detecting defects...
  ✓ Detected 3 defects
[5/5] Localizing to 3D poses...
  - Position and normal for each defect
```

---

## Test 3: Full Pipeline (Vision → Path → Robot)

**Purpose:** End-to-end test without LLM agent

```bash
/opt/anaconda3/bin/python -m src.test_pipeline
```

**Expected:**
- Detects defects
- Generates spiral path (20 waypoints)
- Robot arm moves to approach point
- Executes spiral motion
- First defect turns grey (repaired)

---

## Test 4: LLM Agent

**Purpose:** Test LangGraph agent with Qwen3

```bash
# Make sure Ollama is running
ollama serve &

# Run agent test
/opt/anaconda3/bin/python -m src.test_agent
```

**Expected Output:**
```
[1/3] Creating agent with Qwen3...
  ✓ Agent ready
[2/3] Classifying 3 defects...
  (waits 30-60s per defect)
[3/3] Results:
  Defect 0: RUST
    Severity: moderate
    Strategy: spiral
    Tool: sanding_disc_80
```

**Note:** Takes 60-180 seconds total. Use dashboard with fallback mode for faster demos.

---

## Test 5: Streamlit Dashboard

**Purpose:** Interactive UI with all features

```bash
/opt/anaconda3/bin/streamlit run app/streamlit_app.py
```

**Steps in Browser (http://localhost:8501):**

| Step | Action | Result |
|------|--------|--------|
| 1 | Click **🔄 Initialize** | PyBullet window opens |
| 2 | Click **🔍 Scan** | Defects detected, shown in UI |
| 3 | Click **📋 Plan** | Repair strategies generated |
| 4 | Check **✅ Approve Plan** | Enables Execute button |
| 5 | Click **▶️ Execute** | Robot moves, defects turn grey |

---

## Test 6: Headless Mode (No GUI)

**Purpose:** Run simulation without display (for CI/server)

```bash
/opt/anaconda3/bin/python -c "
from src.simulation.environment import create_environment
from src.simulation.defects import spawn_demo_defects
from src.vision.camera import Camera
from src.vision.detector import DefectDetector

env = create_environment(gui=False)
defects = spawn_demo_defects(env.workpiece_position, env.workpiece_size)
print(f'Created {len(defects)} defects')

camera = Camera(env)
frame = camera.capture()
detector = DefectDetector()
detected = detector.detect(frame['rgb'])
print(f'Detected {len(detected)} defects')

env.close()
print('Headless test passed!')
"
```

---

## Test 7: Individual Modules

### Config Loading
```bash
/opt/anaconda3/bin/python -c "
from src.config import config
print('Config loaded:', config['agent']['model'])
"
```

### Path Generation
```bash
/opt/anaconda3/bin/python -c "
from src.planning.paths import PathGenerator
from src.vision.localization import Pose3D

gen = PathGenerator()
pose = Pose3D((0.5, 0.0, 0.3), (0, 0, 0, 1), (0, 0, 1))
spiral = gen.generate_spiral(pose)
print(f'Spiral: {len(spiral)} waypoints')
"
```

### TSP Ordering
```bash
/opt/anaconda3/bin/python -c "
from src.planning.tsp import optimize_defect_order
from src.agent.tools import DefectInfo

defects = [
    DefectInfo(0, 'rust', (0.7, 0.2, 0.3), 4.0, 0.9),
    DefectInfo(1, 'crack', (0.3, -0.1, 0.3), 2.0, 0.8),
    DefectInfo(2, 'dent', (0.5, 0.0, 0.3), 6.0, 0.85),
]
ordered = optimize_defect_order(defects)
print('Order:', [d.index for d in ordered])
"
```

---

## Test 8: With OpenAI GPT-4o

**Purpose:** Test with production LLM

```bash
# Set API key
export OPENAI_API_KEY=sk-your-key-here

# Update config
sed -i '' 's/provider: ollama/provider: openai/' config.yaml
sed -i '' 's/model: qwen3:14b/model: gpt-4o/' config.yaml

# Run test
/opt/anaconda3/bin/python -m src.test_agent
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: pybullet` | Use `/opt/anaconda3/bin/python` |
| Ollama timeout | Run `ollama serve` first |
| PyBullet window blank | Check GPU drivers |
| Streamlit port in use | `streamlit run app/streamlit_app.py --server.port 8502` |
