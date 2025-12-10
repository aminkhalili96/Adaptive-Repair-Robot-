# 3D Vision: RGBD Depth Analysis

> Point cloud-based geometric defect detection using depth data

---

## Overview

The 3D Vision module enhances defect detection beyond 2D color analysis by using **depth data** to detect geometric anomalies:

| Detection Type | Method | Detects |
|---------------|--------|---------|
| **Color-based** | HSV thresholding | Rust, visible cracks, color anomalies |
| **Geometry-based** | Curvature analysis | Dents, bumps, surface wear, depth discontinuities |

---

## Architecture

```
RGB + Depth → Point Cloud (XYZRGB)
    ↓
Normal Estimation (PCA k-NN)
    ↓
Curvature Computation
    ↓
High Curvature Regions
    ↓
DBSCAN Clustering → Geometric Defects
```

---

## Usage

### In Code

```python
from src.vision.detector import DefectDetector

# Enable hybrid detection
detector = DefectDetector(enable_depth=True)

# Pass both RGB and depth
defects = detector.detect(rgb_image, depth_image, camera_intrinsics)
```

### Via Agent Chat

Ask the chatbot:
- "Perform 3D surface analysis"
- "Scan for geometric defects"
- "Run depth analysis on the part"

---

## Configuration

In `config.py` or `config.yaml`:

```yaml
vision:
  enable_depth_analysis: true
  pointcloud_voxel_size: 0.005    # Meters (smaller = more detail)
  normal_estimation_k: 20          # Neighbors for normal estimation
  curvature_threshold: 0.03        # Above this = defect candidate
  anomaly_clustering_eps: 0.015    # DBSCAN epsilon
  min_defect_points: 30            # Minimum cluster size
```

---

## API Reference

### DepthAnalyzer

```python
from src.vision.depth_analyzer import DepthAnalyzer

analyzer = DepthAnalyzer(
    voxel_size=0.005,
    curvature_threshold=0.03,
    min_defect_points=30
)

# Convert RGBD to point cloud
pointcloud = analyzer.depth_to_pointcloud(rgb, depth, intrinsics)

# Estimate surface normals
normals = analyzer.estimate_normals(pointcloud)

# Compute curvature
curvatures = analyzer.compute_curvature(pointcloud, normals)

# Full detection pipeline
defects = analyzer.detect_geometric_defects(rgb, depth, intrinsics)
```

### GeometricDefect

```python
@dataclass
class GeometricDefect:
    type: GeometricDefectType  # DENT, BUMP, CRACK, SURFACE_WEAR
    position_3d: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    area_m2: float
    curvature: float
    confidence: float
```

---

## Defect Types

| Type | Detection Criteria |
|------|-------------------|
| **DENT** | Concave region (negative curvature) |
| **BUMP** | Convex protrusion (positive curvature) |
| **CRACK** | Depth discontinuity, high variation |
| **SURFACE_WEAR** | Irregular surface texture |

---

## Dependencies

- `scikit-learn` (required) - DBSCAN, k-NN
- `open3d` (optional) - Faster point cloud processing

If `open3d` is not installed, falls back to numpy-based implementation.

---

## Testing

```bash
python -m pytest src/test_depth_analyzer.py -v
```

**17 tests** covering:
- Point cloud conversion
- Normal estimation (unit vectors)
- Curvature computation (flat vs curved)
- Edge cases (empty depth, small images)
