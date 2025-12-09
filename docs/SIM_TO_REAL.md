# Simulation-to-Real Deployment

This document discusses the considerations for deploying AARR from simulation to real hardware.

---

## The Reality Gap

Simulation provides a safe development environment, but real-world deployment introduces:

| Factor | Simulation | Real World |
|--------|------------|------------|
| Physics | Idealized | Imperfect, noisy |
| Sensors | Perfect data | Noise, calibration drift |
| Lighting | Uniform | Variable, shadows |
| Timing | Deterministic | Variable latency |
| Safety | Reset button | Real consequences |

---

## 1. Hand-Eye Calibration

### The Problem

In simulation, we know exactly where the camera is relative to the robot. In reality, we need to calibrate this relationship.

### Solution: Eye-in-Hand Calibration

```
         ┌─────────────┐
         │   Camera    │
         │  (Eye)      │
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │ End Effector│
         │  (Hand)     │
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │  Robot Base │
         └─────────────┘
```

**Procedure:**
1. Mount camera on robot end-effector
2. Place calibration target (checkerboard) in workspace
3. Move robot to multiple poses, capturing images
4. Solve AX = XB problem for camera-to-hand transform
5. Store calibration result

**Libraries:** OpenCV `calibrateHandEye()`, VISP

---

## 2. Tool Center Point (TCP) Calibration

### The Problem

The tool offset (from end-effector to actual tool tip) must be known precisely.

### Solution: 4-Point Method

1. Define a fixed point in space
2. Touch the point from 4+ different orientations
3. Solve for the TCP offset

```python
# Result: tool offset vector
tcp_offset = [dx, dy, dz]  # meters
```

---

## 3. Workpiece Registration

### The Problem

The workpiece position in simulation is exactly known. In reality, it may vary.

### Solution: 3-Point Registration

1. Define 3 reference points on the workpiece
2. Touch each point with the robot
3. Compute transformation from CAD model to actual pose

```python
# Result: 4x4 transformation matrix
T_cad_to_world = compute_registration(cad_points, touched_points)
```

---

## 4. Camera Calibration

### Intrinsic Calibration

Remove lens distortion using checkerboard:

```python
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, image_size
)
```

### Adjustments for Real Cameras

```python
# Undistort images before processing
undistorted = cv2.undistort(image, K, dist)
```

---

## 5. Lighting Robustness

### Simulation Assumption

Uniform, consistent lighting.

### Real-World Challenges

- Ambient light changes
- Shadows from robot/operators
- Specular reflections on metal

### Mitigation Strategies

1. **Controlled Lighting**: Ring light on end-effector
2. **Adaptive Thresholds**: Dynamic HSV ranges
3. **Multiple Exposures**: HDR-like capture
4. **Domain Randomization**: Train with varied lighting

---

## 6. Sensor Noise

### Depth Sensor

Real depth sensors (RealSense, Kinect) have noise:

```python
# Apply bilateral filter to reduce noise while preserving edges
depth_filtered = cv2.bilateralFilter(depth, d=5, sigmaColor=75, sigmaSpace=75)
```

### Normal Estimation

Use larger window for averaging:

```python
# Increase smoothing window for real sensors
smoothed_normals = smooth_normals(normals, window=7)  # vs 3 in sim
```

---

## 7. Real-Time Control

### Simulation

- Direct joint position control
- Instant response
- No latency

### Real Robot

- Motion controller (KRC, teach pendant)
- Communication latency (10-100ms)
- Trajectory blending

### Interface Options

| Method | Latency | Flexibility |
|--------|---------|-------------|
| Teach pendant | High | Low |
| KRL programs | Medium | Medium |
| Robot Web Services | Low | High |
| Direct servo (RSI) | Very Low | Very High |

---

## 8. Safety Certification

### Standards

| Standard | Scope |
|----------|-------|
| ISO 10218-1/2 | Industrial robot safety |
| ISO 12100 | Risk assessment |
| ISO 13849 | Safety control systems |
| IEC 62443 | Cybersecurity |

### Requirements

1. **Risk Assessment**: Document all hazards
2. **Safety Functions**: E-stop, light curtains, guards
3. **Validation**: Test all safety systems
4. **Documentation**: Maintain safety manual

---

## 9. Deployment Checklist

### Pre-Deployment

- [ ] Hand-eye calibration complete
- [ ] TCP calibration verified
- [ ] Workpiece registration tested
- [ ] Camera intrinsics calibrated
- [ ] Lighting conditions evaluated
- [ ] Safety assessment completed

### Integration Testing

- [ ] Move robot to known poses
- [ ] Verify vision detection accuracy
- [ ] Test collision avoidance with dummy obstacles
- [ ] Validate human-in-the-loop workflow
- [ ] Emergency stop tested

### Production Readiness

- [ ] Operator training complete
- [ ] Documentation finalized
- [ ] Maintenance schedule defined
- [ ] Spare parts inventory
- [ ] Fallback procedure documented

---

## 10. Recommended Real Hardware

| Component | Recommendation |
|-----------|----------------|
| Robot | KUKA iiwa, UR10e, Fanuc CRX |
| Camera | Intel RealSense D435, Zivid |
| Lighting | Automated Imaging ring light |
| Compute | Industrial PC, NVIDIA Jetson |
| Safety | Pilz, SICK light curtains |

---

## References

1. Siciliano et al., "Robotics: Modelling, Planning and Control"
2. Intel RealSense SDK documentation
3. KUKA.RSI (Robot Sensor Interface) manual
4. ISO 10218-2:2011 Industrial robots safety
