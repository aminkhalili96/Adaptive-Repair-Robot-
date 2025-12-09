# Safety Architecture

This document outlines the safety mechanisms in the AARR system.

---

## Safety Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     HUMAN OPERATOR                          │
│                   (Final Approval)                          │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                  LLM AGENT LAYER                            │
│           (Reasoning + Fallback Logic)                      │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│               SOFTWARE SAFETY LAYER                          │
│     - Workspace bounds check                                │
│     - Collision detection                                   │
│     - Singularity avoidance                                 │
│     - Input validation                                      │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│               SIMULATION PHYSICS                             │
│               (PyBullet engine)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Workspace Bounds

Every target position is validated against the robot's workspace:

```python
workspace_bounds = {
    "x": [0.2, 0.8],   # meters
    "y": [-0.4, 0.4],  # meters
    "z": [0.05, 0.6],  # meters
}

def check_workspace_bounds(position):
    x, y, z = position
    return (
        bounds["x"][0] <= x <= bounds["x"][1] and
        bounds["y"][0] <= y <= bounds["y"][1] and
        bounds["z"][0] <= z <= bounds["z"][1]
    )
```

**Action if violated**: Motion command rejected, error logged.

---

## 2. Collision Detection

Before and during motion, we check for collisions with the workpiece:

```python
def check_collision(robot_id, obstacle_ids, threshold=0.01):
    for obs_id in obstacle_ids:
        contacts = p.getClosestPoints(
            robot_id, obs_id,
            distance=threshold  # 1cm buffer
        )
        if contacts:
            return True  # Collision detected
    return False
```

### Collision Check Frequency

Per Codex feedback, we check every N waypoints (not just at endpoints):

```python
collision_check_step = 5  # Check every 5th waypoint

for i, waypoint in enumerate(path):
    if i % collision_check_step == 0:
        if check_collision():
            abort_motion()
```

---

## 3. Human-in-the-Loop

The system requires explicit human approval before execution:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   DETECT    │───▶│    PLAN     │───▶│   APPROVE   │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
                                      ┌──────▼──────┐
                                      │   EXECUTE   │
                                      └─────────────┘
```

### Implementation

```python
# Streamlit UI
approved = st.checkbox("✅ Approve Plan")

if st.button("Execute", disabled=not approved):
    execute_repairs()
```

### Rationale

1. **LLM Uncertainty**: AI reasoning may be incorrect
2. **Simulation Gap**: Real-world conditions may differ
3. **Regulatory**: Industrial robots require human oversight
4. **Liability**: Human makes final decision

---

## 4. LLM Fallback

When the LLM fails, we use deterministic fallback strategies:

| Failure Mode | Response |
|--------------|----------|
| Timeout (>30s) | Use fallback strategy |
| Parse error | Use fallback strategy |
| Invalid values | Use fallback strategy |
| 3 consecutive failures | Use fallback, log warning |

```python
FALLBACK_STRATEGIES = {
    "rust": {"strategy": "spiral", "tool": "sanding_disc_80"},
    "crack": {"strategy": "raster", "tool": "filler_applicator"},
    "dent": {"strategy": "circular", "tool": "body_hammer"},
    "unknown": {"strategy": "spiral", "tool": "inspection_only"},
}
```

---

## 5. Input Validation

All external inputs are validated:

### Detection Confidence Threshold

```python
MIN_CONFIDENCE = 0.5  # 50%

def should_flag_for_review(defect):
    if defect.confidence < MIN_CONFIDENCE:
        return True  # Flag for manual review
    return False
```

### Position Sanity Check

```python
def validate_position(pos):
    x, y, z = pos
    # Check for NaN/Inf
    if not all(np.isfinite(p) for p in pos):
        return False
    # Check reasonable bounds
    if abs(x) > 10 or abs(y) > 10 or abs(z) > 10:
        return False
    return True
```

---

## 6. Velocity Limits

Path execution respects velocity and acceleration limits:

```python
path_config = {
    "max_velocity": 0.1,       # m/s
    "max_acceleration": 0.05,  # m/s²
}
```

### Trapezoidal Profile

```
Velocity
   ▲
   │    ┌────────────┐
   │   /              \
   │  /                \
   │ /                  \
   └─────────────────────▶ Time
     Accel   Cruise   Decel
```

---

## 7. Emergency Stop

In simulation, we can terminate at any point:

```python
def emergency_stop():
    # Stop all joint motors
    for i in range(num_joints):
        p.setJointMotorControl2(
            robot_id, i,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=1000
        )
    
    log("EMERGENCY STOP TRIGGERED")
```

---

## 8. Logging & Audit

All safety-related events are logged:

```python
import logging

safety_logger = logging.getLogger("safety")

def log_safety_event(event_type, details):
    safety_logger.warning(
        f"SAFETY: {event_type} - {json.dumps(details)}"
    )

# Example
log_safety_event("WORKSPACE_VIOLATION", {
    "position": [0.9, 0.0, 0.3],
    "bounds": workspace_bounds,
    "action": "REJECTED"
})
```

---

## 9. Known Limitations

| Limitation | Mitigation |
|------------|------------|
| No force/torque sensing | Simulation only |
| No swept-volume collision | Check every N waypoints |
| Singularity detection basic | Simple Jacobian check |
| No external sensor fusion | Future enhancement |

---

## 10. Sim-to-Real Considerations

For real-world deployment, additional safety measures needed:

1. **Hardware E-Stop**: Physical emergency stop button
2. **Light Curtains**: Safety zone monitoring
3. **Force Limiting**: Collaborative robot compliance
4. **Dual-Channel Safety**: Redundant safety controllers
5. **Risk Assessment**: Per ISO 12100 and ISO 10218

See `SIM_TO_REAL.md` for detailed deployment considerations.
