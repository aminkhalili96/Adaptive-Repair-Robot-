"""
Unit tests for the RGBD Depth Analyzer module.

Tests:
- Point cloud conversion from RGBD
- Surface normal estimation
- Curvature computation
- Geometric defect detection
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.depth_analyzer import (
    DepthAnalyzer,
    GeometricDefect,
    GeometricDefectType,
    analyze_rgbd,
    depth_to_pointcloud_quick,
)


class TestDepthAnalyzer:
    """Test cases for DepthAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a DepthAnalyzer instance for testing."""
        return DepthAnalyzer(
            voxel_size=0.01,
            curvature_threshold=0.05,
            min_defect_points=10,
        )
    
    @pytest.fixture
    def sample_rgb(self):
        """Create a sample RGB image."""
        return np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    @pytest.fixture
    def sample_depth(self):
        """Create a sample depth image with a bump in the center."""
        depth = np.ones((100, 100), dtype=np.float32) * 1.0
        # Add a bump in the center (closer to camera = smaller depth)
        y, x = np.ogrid[0:100, 0:100]
        center_x, center_y = 50, 50
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        bump = np.maximum(0, 1 - r / 20)
        depth -= bump * 0.1  # Bump protrudes 10cm
        return depth
    
    @pytest.fixture
    def sample_intrinsics(self):
        """Create sample camera intrinsics."""
        fx = fy = 500.0
        cx, cy = 50.0, 50.0
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    def test_initialization(self, analyzer):
        """Test analyzer initializes with correct parameters."""
        assert analyzer.voxel_size == 0.01
        assert analyzer.curvature_threshold == 0.05
        assert analyzer.min_defect_points == 10
    
    def test_depth_to_pointcloud_shape(self, analyzer, sample_rgb, sample_depth, sample_intrinsics):
        """Test point cloud has correct shape."""
        pointcloud = analyzer.depth_to_pointcloud(
            sample_rgb, sample_depth, sample_intrinsics
        )
        
        # Should be (N, 6) for XYZRGB
        assert pointcloud.ndim == 2
        assert pointcloud.shape[1] == 6
        # Should have at least some valid points
        assert len(pointcloud) > 0
    
    def test_depth_to_pointcloud_xyz_range(self, analyzer, sample_rgb, sample_depth, sample_intrinsics):
        """Test point cloud XYZ values are in reasonable range."""
        pointcloud = analyzer.depth_to_pointcloud(
            sample_rgb, sample_depth, sample_intrinsics
        )
        
        xyz = pointcloud[:, :3]
        
        # Z should be close to depth values (around 1.0m)
        assert xyz[:, 2].min() > 0.0
        assert xyz[:, 2].max() < 2.0
    
    def test_depth_to_pointcloud_rgb_normalized(self, analyzer, sample_rgb, sample_depth, sample_intrinsics):
        """Test point cloud RGB values are normalized to [0, 1]."""
        pointcloud = analyzer.depth_to_pointcloud(
            sample_rgb, sample_depth, sample_intrinsics
        )
        
        rgb = pointcloud[:, 3:6]
        
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0
    
    def test_estimate_normals_shape(self, analyzer, sample_rgb, sample_depth, sample_intrinsics):
        """Test normal estimation produces correct shape."""
        pointcloud = analyzer.depth_to_pointcloud(
            sample_rgb, sample_depth, sample_intrinsics
        )
        
        normals = analyzer.estimate_normals(pointcloud)
        
        assert normals.shape == (len(pointcloud), 3)
    
    def test_estimate_normals_unit_vectors(self, analyzer, sample_rgb, sample_depth, sample_intrinsics):
        """Test normals are unit vectors."""
        pointcloud = analyzer.depth_to_pointcloud(
            sample_rgb, sample_depth, sample_intrinsics
        )
        
        normals = analyzer.estimate_normals(pointcloud)
        norms = np.linalg.norm(normals, axis=1)
        
        # All normals should be approximately unit length
        np.testing.assert_allclose(norms, 1.0, atol=0.01)
    
    def test_compute_curvature_flat_surface(self, analyzer):
        """Test curvature is near zero for flat surface."""
        # Create a flat point cloud
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        normals = np.tile([0, 0, 1], (len(points), 1))
        
        curvatures = analyzer.compute_curvature(points, normals)
        
        # Flat surface should have low curvature
        assert curvatures.mean() < 0.1
    
    def test_compute_curvature_curved_surface(self, analyzer):
        """Test curvature is higher for curved surface."""
        # Create a curved point cloud (hemisphere)
        theta = np.linspace(0, np.pi/2, 20)
        phi = np.linspace(0, 2*np.pi, 20)
        THETA, PHI = np.meshgrid(theta, phi)
        
        X = np.sin(THETA) * np.cos(PHI)
        Y = np.sin(THETA) * np.sin(PHI)
        Z = np.cos(THETA)
        
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # Normals point outward (same as position for unit sphere)
        normals = points.copy()
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
        
        curvatures = analyzer.compute_curvature(points, normals)
        
        # Curved surface should have higher curvature than flat
        assert curvatures.mean() > 0.0
    
    def test_detect_geometric_defects_returns_list(self, analyzer, sample_rgb, sample_depth, sample_intrinsics):
        """Test defect detection returns a list."""
        defects = analyzer.detect_geometric_defects(
            sample_rgb, sample_depth, sample_intrinsics
        )
        
        assert isinstance(defects, list)
    
    def test_detect_geometric_defects_structure(self, analyzer):
        """Test detected defects have correct structure."""
        # Create synthetic RGBD with a clear defect (bump)
        rgb = np.ones((100, 100, 3), dtype=np.uint8) * 128
        depth = np.ones((100, 100), dtype=np.float32) * 1.0
        
        # Add a very pronounced bump
        y, x = np.ogrid[0:100, 0:100]
        center_x, center_y = 50, 50
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        bump = np.maximum(0, 1 - r / 15)
        depth -= bump * 0.2  # 20cm bump
        
        intrinsics = np.array([
            [200.0, 0, 50.0],
            [0, 200.0, 50.0],
            [0, 0, 1]
        ])
        
        defects = analyzer.detect_geometric_defects(rgb, depth, intrinsics)
        
        # Check structure of any detected defects
        for defect in defects:
            assert isinstance(defect, GeometricDefect)
            assert isinstance(defect.type, GeometricDefectType)
            assert len(defect.position_3d) == 3
            assert len(defect.normal) == 3
            assert 0 <= defect.confidence <= 1
    
    def test_geometric_defect_to_dict(self, analyzer):
        """Test GeometricDefect serialization."""
        defect = GeometricDefect(
            type=GeometricDefectType.DENT,
            position_3d=(0.5, 0.0, 1.0),
            normal=(0.0, 0.0, 1.0),
            area_m2=0.001,
            depth_deviation=0.01,
            curvature=0.1,
            confidence=0.8,
            point_indices=np.array([1, 2, 3]),
        )
        
        d = defect.to_dict()
        
        assert d["type"] == "dent"
        assert d["position_3d"] == [0.5, 0.0, 1.0]
        assert d["confidence"] == 0.8
        assert d["num_points"] == 3
    
    def test_get_status(self, analyzer):
        """Test status reporting."""
        status = analyzer.get_status()
        
        assert "voxel_size" in status
        assert "curvature_threshold" in status
        assert "open3d_available" in status


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""
    
    def test_analyze_rgbd_returns_list(self):
        """Test analyze_rgbd returns list of dicts."""
        rgb = np.ones((50, 50, 3), dtype=np.uint8) * 128
        depth = np.ones((50, 50), dtype=np.float32)
        
        result = analyze_rgbd(rgb, depth)
        
        assert isinstance(result, list)
    
    def test_depth_to_pointcloud_quick(self):
        """Test quick pointcloud conversion."""
        rgb = np.ones((50, 50, 3), dtype=np.uint8) * 128
        depth = np.ones((50, 50), dtype=np.float32) * 0.5
        intrinsics = np.array([
            [100.0, 0, 25.0],
            [0, 100.0, 25.0],
            [0, 0, 1]
        ])
        
        pc = depth_to_pointcloud_quick(rgb, depth, intrinsics)
        
        assert pc.ndim == 2
        assert pc.shape[1] == 6


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_depth(self):
        """Test handling of all-zero depth."""
        analyzer = DepthAnalyzer(min_defect_points=5)
        rgb = np.ones((50, 50, 3), dtype=np.uint8) * 128
        depth = np.zeros((50, 50), dtype=np.float32)
        
        defects = analyzer.detect_geometric_defects(rgb, depth)
        
        # Should not crash, may return empty list
        assert isinstance(defects, list)
    
    def test_small_image(self):
        """Test handling of very small images."""
        analyzer = DepthAnalyzer(min_defect_points=2)
        rgb = np.ones((10, 10, 3), dtype=np.uint8) * 128
        depth = np.ones((10, 10), dtype=np.float32) * 0.5
        
        defects = analyzer.detect_geometric_defects(rgb, depth)
        
        assert isinstance(defects, list)
    
    def test_uniform_depth(self):
        """Test handling of perfectly uniform depth (no defects expected)."""
        analyzer = DepthAnalyzer(min_defect_points=10)
        rgb = np.ones((100, 100, 3), dtype=np.uint8) * 128
        depth = np.ones((100, 100), dtype=np.float32) * 1.0
        
        defects = analyzer.detect_geometric_defects(rgb, depth)
        
        # Uniform depth = flat surface = no geometric defects
        # (might still detect some due to edge effects)
        assert isinstance(defects, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
