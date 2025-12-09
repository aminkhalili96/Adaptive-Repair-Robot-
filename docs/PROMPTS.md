# LLM Agent Prompts

This document details the prompting strategy for the AARR repair agent.

---

## System Prompt

The agent uses a carefully crafted system prompt that establishes its role as an industrial Process Engineer:

```
You are an expert Process Engineer for industrial surface repair in MRO 
(Maintenance, Repair, and Overhaul) operations.

Your role is to analyze defect data and determine optimal repair parameters.
```

### Key Design Decisions

1. **Domain Expert Persona**: Established as a "Process Engineer" not just "AI assistant"
2. **Structured Output**: Explicitly requires JSON format for reliable parsing
3. **Decision Guidelines**: Provides clear rules for each defect type
4. **Safety Emphasis**: Includes workspace bounds and confidence thresholds

---

## Classification Prompt Template

```python
CLASSIFICATION_PROMPT = """
Classify the severity of this defect and recommend a repair strategy.

Defect Information:
- Type: {defect_type}
- Position: ({x:.3f}, {y:.3f}, {z:.3f}) meters
- Approximate size: {size:.1f} cm²
- Detection confidence: {confidence:.0%}

Respond with JSON only:
{{
    "severity": "minor|moderate|severe",
    "strategy": "spiral|raster|circular",
    "tool": "tool_name",
    "estimated_time_seconds": integer,
    "notes": "brief explanation"
}}
"""
```

### Why This Format?

- **Explicit field types**: Reduces ambiguity in LLM output
- **Constrained choices**: Limits severity/strategy to valid options
- **JSON-only instruction**: Reduces extraneous text

---

## Fallback Strategies

When LLM fails (timeout, parsing error, invalid response), we use rule-based fallbacks:

```python
FALLBACK_STRATEGIES = {
    "rust": {
        "severity": "moderate",
        "strategy": "spiral",
        "tool": "sanding_disc_80",
        "estimated_time_seconds": 30,
    },
    "crack": {
        "severity": "moderate",
        "strategy": "raster",
        "tool": "filler_applicator",
        "estimated_time_seconds": 45,
    },
    "dent": {
        "severity": "moderate",
        "strategy": "circular",
        "tool": "body_hammer",
        "estimated_time_seconds": 60,
    },
    "unknown": {
        "severity": "minor",
        "strategy": "spiral",
        "tool": "inspection_only",
        "estimated_time_seconds": 15,
    },
}
```

### Fallback Triggers

1. **Timeout**: LLM response exceeds 30 seconds
2. **Parse Error**: Response is not valid JSON
3. **Validation Error**: Required fields missing or invalid values
4. **Retry Exhausted**: 3 consecutive failures

---

## Response Validation

All LLM responses are validated before use:

```python
def validate_repair_plan(plan: Dict) -> bool:
    # Required fields
    required = ["severity", "strategy", "tool", "estimated_time_seconds"]
    for field in required:
        if field not in plan:
            return False
    
    # Value constraints
    if plan["severity"] not in ["minor", "moderate", "severe"]:
        return False
    
    if plan["strategy"] not in ["spiral", "raster", "circular"]:
        return False
    
    # Time bounds (sanity check)
    if plan["estimated_time_seconds"] <= 0:
        return False
    if plan["estimated_time_seconds"] > 300:
        return False
    
    return True
```

---

## Retry Strategy

```python
for attempt in range(max_retries):  # Default: 3
    try:
        response = await asyncio.wait_for(
            call_llm(prompt),
            timeout=30  # seconds
        )
        
        plan = parse_json(response)
        if validate(plan):
            return plan  # Success!
            
    except TimeoutError:
        log(f"Attempt {attempt + 1}: Timeout")
    except Exception as e:
        log(f"Attempt {attempt + 1}: {e}")

# All retries failed
return get_fallback_plan(defect_type)
```

---

## Example LLM Interaction

### Input
```
Defect Information:
- Type: rust
- Position: (0.550, 0.050, 0.260) meters
- Approximate size: 4.5 cm²
- Detection confidence: 92%
```

### Expected Output
```json
{
    "severity": "moderate",
    "strategy": "spiral",
    "tool": "sanding_disc_80",
    "estimated_time_seconds": 35,
    "notes": "Medium rust patch, spiral sanding recommended for even coverage"
}
```

---

## Model Configuration

```yaml
agent:
  provider: ollama      # or "openai"
  model: qwen3:14b      # or "gpt-4o"
  temperature: 0.1      # Low for consistency
  timeout: 30           # seconds
  max_retries: 3
```

### Why Low Temperature?

- Industrial applications need **consistent**, predictable responses
- Creative variation could lead to unexpected tool selections
- 0.1 provides slight flexibility while maintaining reliability

---

## Future Improvements

1. **Chain-of-Thought**: Add reasoning steps before final answer
2. **Multi-turn Context**: Include repair history for similar defects
3. **Confidence Calibration**: Train on labeled dataset for severity
4. **Tool Recommendation**: Fine-tune on material-specific strategies
