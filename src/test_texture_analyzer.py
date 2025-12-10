"""
Unit tests for the Texture Analyzer module.

Tests:
- UV coordinate generation (planar, cylindrical)
- Procedural texture generation
- Defect detection from textures
- 2D-to-3D mapping
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.texture_analyzer import (
    TextureAnalyzer,
    TextureDefect,
    scan_mesh_for_defects,
)


class TestTextureAnalyzer:
    """Test cases for TextureAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        return TextureAnalyzer()
    
    @pytest.fixture
    def sample_vertices(self):
        """Create a grid of vertices."""
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    def test_generate_uv_planar(self, analyzer, sample_vertices):
        """Test planar UV projection."""
        uvs = analyzer.generate_uv_coords(sample_vertices, method="planar")
        
        assert uvs.shape == (100, 2)
        assert uvs.min() >= 0
        assert uvs.max() <= 1
    
    def test_generate_uv_cylindrical(self, analyzer):
        """Test cylindrical UV projection."""
        # Create cylinder-like points
        theta = np.linspace(0, 2*np.pi, 20)
        z = np.linspace(0, 1, 10)
        THETA, Z = np.meshgrid(theta, z)
        X = np.cos(THETA)
        Y = np.sin(THETA)
        vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        uvs = analyzer.generate_uv_coords(vertices, method="cylindrical")
        
        assert uvs.shape[1] == 2
        assert uvs.min() >= 0
        assert uvs.max() <= 1
    
    def test_generate_rust_texture_shape(self, analyzer):
        """Test texture generation produces correct shape."""
        texture = analyzer.generate_rust_texture(256, 256, n_rust_spots=2)
        
        assert texture.shape == (256, 256, 3)
        assert texture.dtype == np.uint8
    
    def test_generate_rust_texture_reproducible(self, analyzer):
        """Test texture generation is reproducible with seed."""
        tex1 = analyzer.generate_rust_texture(128, 128, seed=42)
        tex2 = analyzer.generate_rust_texture(128, 128, seed=42)
        
        np.testing.assert_array_equal(tex1, tex2)
    
    def test_detect_defects_clean_surface(self, analyzer):
        """Test no defects detected on gray surface."""
        # Pure gray texture (no rust)
        texture = np.full((100, 100, 3), 128, dtype=np.uint8)
        
        mask, coverage = analyzer.detect_defects_from_texture(texture)
        
        assert mask.shape == (100, 100)
        assert coverage < 0.01  # Near zero coverage
    
    def test_detect_defects_rust_surface(self, analyzer):
        """Test rust detection on rusty texture."""
        # Texture with red-brown rust color
        texture = np.full((100, 100, 3), [180, 50, 50], dtype=np.uint8)  # Rust color
        
        mask, coverage = analyzer.detect_defects_from_texture(texture)
        
        assert mask.shape == (100, 100)
        assert coverage > 0.5  # Should detect most of it
    
    def test_map_mask_to_3d(self, analyzer, sample_vertices):
        """Test mapping 2D mask to 3D vertices."""
        uvs = analyzer.generate_uv_coords(sample_vertices, method="planar")
        
        # Create mask with defect in center
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 255  # Center square
        
        indices = analyzer.map_mask_to_3d(mask, sample_vertices, uvs)
        
        assert len(indices) > 0
        assert all(isinstance(i, (int, np.integer)) for i in indices)
    
    def test_analyze_texture_full_pipeline(self, analyzer, sample_vertices):
        """Test full analysis pipeline."""
        uvs = analyzer.generate_uv_coords(sample_vertices, method="planar")
        texture = analyzer.generate_rust_texture(256, 256, n_rust_spots=3, seed=42)
        
        defects = analyzer.analyze_texture(texture, sample_vertices, uvs)
        
        assert isinstance(defects, list)
        for d in defects:
            assert isinstance(d, TextureDefect)
            assert len(d.position_3d) == 3
            assert 0 <= d.confidence <= 1
    
    def test_get_vertex_colors(self, analyzer, sample_vertices):
        """Test vertex color sampling from texture."""
        uvs = analyzer.generate_uv_coords(sample_vertices, method="planar")
        texture = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8)
        
        colors = analyzer.get_vertex_colors(sample_vertices, uvs, texture)
        
        assert colors.shape == (len(sample_vertices), 3)
        # All colors should match the uniform texture
        np.testing.assert_array_equal(colors[0], [100, 150, 200])


class TestScanMeshForDefects:
    """Test the high-level scan function."""
    
    def test_scan_returns_aarr_format(self):
        """Test scan returns AARR-compatible defect format."""
        # Simple grid mesh
        x = np.linspace(-1, 1, 20)
        y = np.linspace(-1, 1, 20)
        X, Y = np.meshgrid(x, y)
        vertices = np.column_stack([X.ravel(), Y.ravel(), np.zeros(400)])
        
        defects = scan_mesh_for_defects(vertices)
        
        assert isinstance(defects, list)
        for d in defects:
            assert "position" in d
            assert "type" in d
            assert "severity" in d
            assert "confidence" in d
            assert d["source"] == "texture_analysis"


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_vertices(self):
        """Test handling of empty vertex array."""
        analyzer = TextureAnalyzer()
        vertices = np.zeros((0, 3))
        uvs = analyzer.generate_uv_coords(vertices, method="planar")
        
        assert uvs.shape == (0, 2)
    
    def test_single_vertex(self):
        """Test handling of single vertex."""
        analyzer = TextureAnalyzer()
        vertices = np.array([[0.5, 0.5, 0.0]])
        
        # Should not crash
        uvs = analyzer.generate_uv_coords(vertices, method="planar")
        assert uvs.shape == (1, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
