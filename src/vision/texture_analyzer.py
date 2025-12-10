"""
Texture-to-3D Defect Analyzer.

This module enables a real "Scan-to-Path" workflow by:
1. Generating UV coordinates for mesh vertices
2. Creating procedural texture maps with defect patterns
3. Detecting defects from texture images using CV
4. Mapping 2D texture defects back to 3D vertex positions

This makes the defect detection pipeline "real" instead of using hardcoded metadata.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextureDefect:
    """A defect detected from texture analysis and mapped to 3D."""
    type: str
    uv_center: Tuple[float, float]  # UV coordinates (0-1)
    position_3d: Tuple[float, float, float]  # World coordinates
    area_uv: float  # Area in UV space
    coverage_percent: float
    vertex_indices: np.ndarray  # Indices of affected vertices
    confidence: float


class TextureAnalyzer:
    """
    Analyzes textures applying to 3D meshes to detect defects.
    
    Workflow:
    1. Generate UV coordinates for mesh vertices
    2. Create or load a texture image
    3. Run CV detection on texture
    4. Map detected regions back to 3D vertices
    
    Example:
        >>> analyzer = TextureAnalyzer()
        >>> uvs = analyzer.generate_uv_coords(vertices)
        >>> texture = analyzer.generate_rust_texture(256, 256)
        >>> defects = analyzer.analyze_texture(texture, vertices, uvs)
    """
    
    def __init__(
        self,
        rust_hsv_lower: Tuple[int, int, int] = (0, 100, 100),
        rust_hsv_upper: Tuple[int, int, int] = (10, 255, 255),
        min_defect_area: float = 0.001,
    ):
        """
        Initialize the texture analyzer.
        
        Args:
            rust_hsv_lower: Lower HSV bound for rust detection
            rust_hsv_upper: Upper HSV bound for rust detection
            min_defect_area: Minimum defect area as fraction of total texture
        """
        self.rust_hsv_lower = np.array(rust_hsv_lower)
        self.rust_hsv_upper = np.array(rust_hsv_upper)
        self.min_defect_area = min_defect_area
    
    def generate_uv_coords(
        self,
        vertices: np.ndarray,
        method: str = "planar"
    ) -> np.ndarray:
        """
        Generate UV coordinates for mesh vertices.
        
        Args:
            vertices: (N, 3) array of vertex positions
            method: UV mapping method ("planar", "cylindrical", "spherical")
            
        Returns:
            (N, 2) array of UV coordinates in [0, 1] range
        """
        n_verts = len(vertices)
        
        # Handle empty vertices
        if n_verts == 0:
            return np.zeros((0, 2))
        
        if method == "planar":
            # Planar projection: X, Y -> U, V
            x, y = vertices[:, 0], vertices[:, 1]
            u = (x - x.min()) / (x.max() - x.min() + 1e-8)
            v = (y - y.min()) / (y.max() - y.min() + 1e-8)
        
        elif method == "cylindrical":
            # Cylindrical projection: angle -> U, height -> V
            x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
            theta = np.arctan2(y, x)
            u = (theta + np.pi) / (2 * np.pi)
            v = (z - z.min()) / (z.max() - z.min() + 1e-8)
        
        elif method == "spherical":
            # Spherical projection
            x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2) + 1e-8
            theta = np.arctan2(y, x)
            phi = np.arccos(np.clip(z / r, -1, 1))
            u = (theta + np.pi) / (2 * np.pi)
            v = phi / np.pi
        
        else:
            raise ValueError(f"Unknown UV method: {method}")
        
        return np.column_stack([u, v])
    
    def generate_rust_texture(
        self,
        width: int = 512,
        height: int = 512,
        n_rust_spots: int = 3,
        rust_radius_range: Tuple[float, float] = (0.05, 0.15),
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate a procedural texture with rust spots.
        
        Args:
            width: Texture width in pixels
            height: Texture height in pixels
            n_rust_spots: Number of rust spots to generate
            rust_radius_range: Min/max radius as fraction of image size
            seed: Random seed for reproducibility
            
        Returns:
            (H, W, 3) uint8 RGB texture image
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Base: Metallic gray
        texture = np.full((height, width, 3), [192, 192, 192], dtype=np.uint8)
        
        # Add subtle brushed metal pattern
        noise = np.random.normal(0, 10, (height, width)).astype(np.int16)
        for c in range(3):
            texture[:, :, c] = np.clip(
                texture[:, :, c].astype(np.int16) + noise,
                0, 255
            ).astype(np.uint8)
        
        # Add rust spots
        for _ in range(n_rust_spots):
            cx = np.random.uniform(0.1, 0.9) * width
            cy = np.random.uniform(0.1, 0.9) * height
            radius = np.random.uniform(*rust_radius_range) * min(width, height)
            
            # Create gradient mask for soft edges
            y, x = np.ogrid[:height, :width]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            mask = np.clip(1 - dist / radius, 0, 1)
            
            # Rust colors (red-brown gradient)
            rust_colors = [
                [139, 69, 19],   # Saddle brown
                [165, 42, 42],   # Brown
                [178, 34, 34],   # Firebrick
                [205, 92, 0],    # Orange-red
            ]
            rust_color = np.array(rust_colors[np.random.randint(len(rust_colors))])
            
            # Apply rust with soft blending
            for c in range(3):
                texture[:, :, c] = (
                    texture[:, :, c] * (1 - mask * 0.9) +
                    rust_color[c] * mask * 0.9
                ).astype(np.uint8)
        
        return texture
    
    def detect_defects_from_texture(
        self,
        texture: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Detect defects in a texture image using CV.
        
        Args:
            texture: (H, W, 3) uint8 RGB texture image
            
        Returns:
            mask: (H, W) binary mask (255 = defect, 0 = clean)
            coverage: Fraction of texture covered by defects
        """
        # Convert to HSV for color thresholding
        hsv = cv2.cvtColor(texture, cv2.COLOR_RGB2HSV)
        
        # Detect rust (red-brown colors)
        mask1 = cv2.inRange(hsv, self.rust_hsv_lower, self.rust_hsv_upper)
        
        # Also check red wraparound (H: 170-180)
        mask2 = cv2.inRange(
            hsv,
            np.array([170, 100, 100]),
            np.array([180, 255, 255])
        )
        
        # Also detect brown (H: 10-20)
        mask3 = cv2.inRange(
            hsv,
            np.array([10, 100, 50]),
            np.array([30, 255, 200])
        )
        
        # Combine masks
        mask = mask1 | mask2 | mask3
        
        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate coverage
        total_pixels = mask.shape[0] * mask.shape[1]
        defect_pixels = np.sum(mask > 0)
        coverage = defect_pixels / total_pixels
        
        return mask, coverage
    
    def map_mask_to_3d(
        self,
        mask: np.ndarray,
        vertices: np.ndarray,
        uv_coords: np.ndarray
    ) -> List[int]:
        """
        Map a 2D defect mask to 3D vertex indices.
        
        For each vertex, look up its UV coordinate in the mask.
        If the mask pixel is > 0, mark that vertex as defective.
        
        Args:
            mask: (H, W) binary defect mask
            vertices: (N, 3) vertex positions
            uv_coords: (N, 2) UV coordinates for each vertex
            
        Returns:
            List of vertex indices that are in defect regions
        """
        h, w = mask.shape
        defect_indices = []
        
        for i, (u, v) in enumerate(uv_coords):
            # Convert UV to pixel coordinates
            px = int(np.clip(u * (w - 1), 0, w - 1))
            py = int(np.clip(v * (h - 1), 0, h - 1))
            
            # Check if this vertex maps to a defect pixel
            if mask[py, px] > 0:
                defect_indices.append(i)
        
        return defect_indices
    
    def analyze_texture(
        self,
        texture: np.ndarray,
        vertices: np.ndarray,
        uv_coords: np.ndarray
    ) -> List[TextureDefect]:
        """
        Full pipeline: detect defects from texture and map to 3D.
        
        Args:
            texture: (H, W, 3) RGB texture image
            vertices: (N, 3) mesh vertices
            uv_coords: (N, 2) UV coordinates
            
        Returns:
            List of TextureDefect objects with 3D positions
        """
        # Detect defects in texture
        mask, coverage = self.detect_defects_from_texture(texture)
        
        if coverage < self.min_defect_area:
            return []
        
        # Find connected components (separate defect regions)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        h, w = mask.shape
        defects = []
        
        # Skip label 0 (background)
        for label_id in range(1, num_labels):
            # Get region stats
            area_px = stats[label_id, cv2.CC_STAT_AREA]
            area_uv = area_px / (h * w)
            
            if area_uv < self.min_defect_area:
                continue
            
            # Get UV center from centroid
            cx, cy = centroids[label_id]
            uv_center = (cx / w, cy / h)
            
            # Find vertices in this region
            region_mask = (labels == label_id).astype(np.uint8) * 255
            vertex_indices = self.map_mask_to_3d(region_mask, vertices, uv_coords)
            
            if len(vertex_indices) == 0:
                continue
            
            # Calculate 3D position (centroid of affected vertices)
            affected_verts = vertices[vertex_indices]
            position_3d = tuple(affected_verts.mean(axis=0))
            
            # Confidence based on area
            confidence = min(1.0, area_uv / 0.05)
            
            defects.append(TextureDefect(
                type="rust",
                uv_center=uv_center,
                position_3d=position_3d,
                area_uv=area_uv,
                coverage_percent=area_uv * 100,
                vertex_indices=np.array(vertex_indices),
                confidence=confidence,
            ))
        
        return defects
    
    def get_vertex_colors(
        self,
        vertices: np.ndarray,
        uv_coords: np.ndarray,
        texture: np.ndarray
    ) -> np.ndarray:
        """
        Sample texture colors for each vertex based on UV coords.
        
        This enables texture-based vertex coloring in Plotly.
        
        Args:
            vertices: (N, 3) vertex positions
            uv_coords: (N, 2) UV coordinates
            texture: (H, W, 3) RGB texture
            
        Returns:
            (N, 3) RGB colors for each vertex
        """
        h, w = texture.shape[:2]
        n_verts = len(vertices)
        colors = np.zeros((n_verts, 3), dtype=np.uint8)
        
        for i, (u, v) in enumerate(uv_coords):
            px = int(np.clip(u * (w - 1), 0, w - 1))
            py = int(np.clip(v * (h - 1), 0, h - 1))
            colors[i] = texture[py, px]
        
        return colors


# ============ HELPER FUNCTIONS ============

def create_textured_mesh_trace(
    vertices: np.ndarray,
    faces: np.ndarray,
    texture: np.ndarray,
    uv_method: str = "planar"
):
    """
    Create a Plotly Mesh3d trace with texture-based vertex colors.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) face indices
        texture: (H, W, 3) RGB texture image
        uv_method: UV mapping method
        
    Returns:
        go.Mesh3d trace
    """
    import plotly.graph_objects as go
    
    analyzer = TextureAnalyzer()
    uv_coords = analyzer.generate_uv_coords(vertices, method=uv_method)
    vertex_colors = analyzer.get_vertex_colors(vertices, uv_coords, texture)
    
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        vertexcolor=vertex_colors,
        flatshading=False,
        lighting={
            'ambient': 0.3,
            'diffuse': 0.6,
            'roughness': 0.1,
            'specular': 0.5,
            'fresnel': 0.2
        },
        lightposition={'x': 100, 'y': 100, 'z': 2000}
    )


def scan_mesh_for_defects(
    vertices: np.ndarray,
    texture: Optional[np.ndarray] = None,
    uv_method: str = "planar"
) -> List[Dict]:
    """
    High-level function: scan a mesh for defects using texture analysis.
    
    If no texture provided, generates a procedural one with rust spots.
    
    Returns:
        List of defect dictionaries compatible with existing AARR format
    """
    analyzer = TextureAnalyzer()
    uv_coords = analyzer.generate_uv_coords(vertices, method=uv_method)
    
    if texture is None:
        texture = analyzer.generate_rust_texture(512, 512, n_rust_spots=3)
    
    texture_defects = analyzer.analyze_texture(texture, vertices, uv_coords)
    
    # Convert to AARR format
    return [
        {
            "position": td.position_3d,
            "type": td.type,
            "severity": "high" if td.coverage_percent > 5 else "medium" if td.coverage_percent > 2 else "low",
            "confidence": td.confidence,
            "coverage_percent": td.coverage_percent,
            "source": "texture_analysis",
            "vertex_count": len(td.vertex_indices),
        }
        for td in texture_defects
    ]
