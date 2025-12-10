# Evaluation & Testing Guide

This document describes the evaluation and testing infrastructure for the AARR project.

---

## Quick Start

```bash
# Run all unit tests
pytest src/ -v

# Run agent evaluation
python src/eval_agent.py

# Run vision evaluation
python src/eval_vision.py

# Run E2E workflow tests
pytest src/test_e2e_workflow.py -v
```

---

## Evaluation Scripts

### 1. Agent Evaluation (`eval_agent.py`)

Tests the LLM agent's ability to:
- **Tool Calling Accuracy**: Correct tools for commands
- **Response Quality**: Relevant keywords in responses
- **Safety Compliance**: Human-in-loop approval handling

**Usage:**
```bash
# Run with mock responses (no API calls)
python src/eval_agent.py

# Run with real OpenAI API
python src/eval_agent.py --real

# Save results to JSON
python src/eval_agent.py --save
```

**Metrics:**
| Metric | Description | Target |
|--------|-------------|--------|
| Tool Accuracy | % correct tools called | ≥80% |
| Keyword Match | % expected keywords in response | ≥50% |
| Pass Rate | Overall test pass rate | ≥80% |

---

### 2. Vision Evaluation (`eval_vision.py`)

Tests the vision pipeline with synthetic defect images:
- **Precision**: How many detections are correct
- **Recall**: How many defects are found
- **IoU**: Mask overlap accuracy

**Usage:**
```bash
# Run with default 10 images
python src/eval_vision.py

# Run with more images
python src/eval_vision.py --n-images 50

# Save results
python src/eval_vision.py --save
```

**Metrics:**
| Metric | Description | Target |
|--------|-------------|--------|
| Precision | TP / (TP + FP) | ≥70% |
| Recall | TP / (TP + FN) | ≥70% |
| F1 Score | Harmonic mean of P & R | ≥70% |
| IoU | Intersection / Union | ≥50% |

---

### 3. E2E Workflow Tests (`test_e2e_workflow.py`)

Comprehensive pytest tests covering:
- Mesh loading & visualization
- Scan → Detect → Plan → Execute pipeline
- Agent chat integration
- Performance benchmarks
- Code interpreter security

**Usage:**
```bash
# Run all E2E tests
pytest src/test_e2e_workflow.py -v

# Run specific test class
pytest src/test_e2e_workflow.py::TestMeshLoading -v

# Run with timing output
pytest src/test_e2e_workflow.py::TestPerformanceBenchmarks -v -s
```

---

## Performance Benchmarks

| Operation | Target Latency | Test |
|-----------|---------------|------|
| Mesh Load | < 500ms | `test_mesh_load_latency` |
| TSP (30 defects) | < 1000ms | `test_tsp_optimization_latency` |
| ML Prediction | < 50ms | `test_ml_prediction_latency` |
| RAG Query | < 100ms | `test_knowledge_base_latency` |

---

## Adding New Tests

### Agent Scenarios
Add to `TEST_SCENARIOS` in `eval_agent.py`:
```python
TestScenario(
    name="your_test_name",
    user_input="User command here",
    expected_tools=["tool_name"],
    expected_keywords=["expected", "words"],
    category="tool_calling"  # or "response" or "safety"
)
```

### Vision Test Cases
Modify `generate_synthetic_dataset()` in `eval_vision.py` to add specific defect patterns.

### E2E Tests
Add pytest test methods to the appropriate test class in `test_e2e_workflow.py`.

---

## Interpreting Results

### Agent Eval
- ✅ **Pass**: Tool accuracy ≥80% AND keyword match ≥50%
- ⚠️ **Warning**: One metric below threshold
- ❌ **Fail**: Both metrics below threshold

### Vision Eval
- ✅ **Good**: F1 ≥0.7 AND IoU ≥0.5
- ⚠️ **OK**: F1 ≥0.5
- ❌ **Needs Work**: F1 <0.5

---

## CI Integration

Add to your CI pipeline:
```yaml
jobs:
  test:
    steps:
      - name: Run Unit Tests
        run: pytest src/ -v --tb=short
      
      - name: Run Agent Eval
        run: python src/eval_agent.py --save
      
      - name: Run Vision Eval
        run: python src/eval_vision.py --save
```
