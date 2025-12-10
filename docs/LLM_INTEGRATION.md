# LLM Integration Guide

> Deep integration of Large Language Models throughout the AARR robotic repair system.

This document covers all LLM-powered features in AARR, including configuration, usage, and API reference.

---

## Overview

AARR uses LLMs for six key capabilities:

| Feature | Module | Description |
|---------|--------|-------------|
| **Multimodal Diagnosis** | `llm_defect_classifier.py` | GPT-4o Vision for structured defect classification |
| **Code Interpreter** | `code_interpreter.py` | LLM-written Python for custom path generation |
| **Quality Reports** | `report_generator.py` | LLM-generated audit-ready documentation |
| **What-If Simulation** | `supervisor_agent.py` | Hypothetical scenario comparison |
| **Memory Bank** | `memory_bank.py` | Vector store for past repair recall |
| **Safety Reviewer** | `safety_reviewer.py` | Second LLM pass for plan validation |

---

## Configuration

All LLM features require an OpenAI API key:

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
```

Alternatively, some features can fall back to local models via Ollama.

---

## Feature 1: Multimodal Defect Diagnosis

### Purpose
Uses GPT-4o Vision to analyze images and return **structured JSON** classifications.

### Usage
```python
from src.vision.llm_defect_classifier import classify_defect_from_image

result = classify_defect_from_image(image_base64)
print(result.get_summary())
```

### Output Structure
```json
{
    "defect_type": "rust",
    "severity": 7,
    "recommended_action": "grinding",
    "confidence": 0.85,
    "location_description": "upper-left quadrant"
}
```

### Chat Tool
Say: **"Classify this defect"** or **"What type of defect is this?"**

---

## Feature 2: Code Interpreter

### Purpose
Allows the LLM to write Python code that generates custom geometric toolpaths.

### Usage
```python
from src.planning.code_interpreter import exec_custom_path

result = exec_custom_path(
    code="def generate_custom_path(center, radius): ...",
    center=(0.5, 0, 0.3),
    radius=0.05
)
```

### Security
- Sandboxed execution with `RestrictedPython`
- Only `numpy` and `math` allowed
- 5-second timeout
- Return type validation

### Chat Tool
Say: **"Generate a star pattern"** or **"Create a zigzag path"**

---

## Feature 3: Quality Reports

### Purpose
Generates audit-ready markdown reports after repairs using LLM formatting.

### Usage
```python
from src.agent.report_generator import generate_quality_report

report = generate_quality_report(
    repair_log={"actions_count": 2, "status": "Completed"},
    defects=[{"type": "rust", "severity": "high"}],
    part_info={"part_id": "TB-001", "material": "Steel"}
)
print(report.to_markdown())
```

### Chat Tool
Say: **"Generate a quality report"** or **"Create documentation"**

---

## Feature 4: What-If Simulation

### Purpose
Allows users to ask hypothetical questions and see comparison tables.

### Example Queries
- "What if the defect was twice as large?"
- "What if this was aluminum instead of steel?"
- "What if we had half as many defects?"

### Output
```
| Metric | Original | Hypothetical |
|--------|----------|--------------|
| Defects | 3 | 3 |
| Est. Time | 180s | 324s |
| Time Diff | - | +144s |
```

### Chat Tool
Say: **"What if..."** or **"Hypothetically..."**

---

## Feature 5: Memory Bank

### Purpose
Stores completed repairs and recalls similar experiences for future reference.

### Usage
```python
from src.agent.memory_bank import store_repair, recall_similar_repairs

# Store a repair
store_repair(
    summary="Ground rust off steel turbine blade",
    part_type="turbine_blade",
    defect_type="rust",
    duration_seconds=180
)

# Recall similar repairs
results = recall_similar_repairs("rust on metal part")
for r in results:
    print(r.get_summary())
```

### Persistence
Memories are stored in `temp/memory_bank.json` and persist across sessions.

### Chat Tool
Say: **"What did we do before?"** or **"Similar past repairs"**

---

## Feature 6: Safety Reviewer

### Purpose
Performs an independent LLM-based safety check before executing repairs.

### Checks Performed
- Workspace bounds (X: 0.2-0.8m, Y: -0.4-0.4m, Z: 0.05-0.6m)
- RPM limits by material (Aluminum: 1500, Steel: 4000, Composite: 1000)
- Tool compatibility
- Path safety

### Usage
```python
from src.agent.safety_reviewer import review_plan_safety

result = review_plan_safety({
    "defects": [...],
    "material": "steel",
    "rpm": 3000
})
print(result.get_summary())
```

### Chat Tool
Say: **"Is this plan safe?"** or **"Review for safety"**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Chat Input                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               Supervisor Agent (GPT-4o)                     │
│                                                             │
│   Tools:                                                    │
│   ├── focus_camera_on_defect                               │
│   ├── trigger_scan                                          │
│   ├── classify_defect_visual ──────► LLM Vision Classifier │
│   ├── generate_custom_path ────────► Code Interpreter       │
│   ├── generate_quality_report ─────► Report Generator       │
│   ├── recall_past_repairs ─────────► Memory Bank            │
│   ├── simulate_scenario ───────────► What-If Engine         │
│   └── review_plan_safety ──────────► Safety Reviewer        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## API Cost Considerations

Each repair cycle may incur the following LLM calls:

| Action | Approximate Tokens |
|--------|-------------------|
| Chat response | ~500-800 |
| Visual classification | ~800-1200 |
| Quality report | ~1500-2000 |
| Safety review | ~500-800 |
| What-if simulation | ~300-500 |

**Estimated cost per repair cycle**: $0.02-0.05 (GPT-4o pricing)

---

## Fallback Behavior

All LLM features have fallback modes when the API is unavailable:

| Feature | Fallback |
|---------|----------|
| Defect Classifier | Returns "unknown" with 0.3 confidence |
| Report Generator | Uses template-based generation |
| Safety Reviewer | Uses heuristic rule checks |
| Memory Bank | Full functionality (local storage) |

---

## Testing

```bash
# Run all LLM integration tests
pytest src/test_llm_classifier.py src/test_memory_bank.py src/test_safety_reviewer.py -v

# Test individual modules
python -m src.vision.llm_defect_classifier
python -m src.agent.memory_bank
python -m src.agent.safety_reviewer
```
