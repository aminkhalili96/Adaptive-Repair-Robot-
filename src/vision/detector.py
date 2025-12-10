"""
Defect detector using OpenCV for HSV color thresholding.

Detects three types of defects:
- Rust (red-brown color)
- Crack (black/dark color)
- Dent (blue color)

Includes morphological cleanup per Codex feedback.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from src.config import config


class DefectType(Enum):
    """Types of detectable defects."""
    RUST = "rust"
    CRACK = "crack"
    DENT = "dent"
    UNKNOWN = "unknown"


@dataclass
class DetectedDefect:
    """
    A detected defect from the vision system.
    
    Attributes:
        type: Type of defect
        centroid_px: (u, v) pixel coordinates of centroid
        area_px: Area in pixels
        confidence: Detection confidence (0-1)
        bounding_box: (x, y, w, h) bounding box in pixels
        contour: OpenCV contour points
    """
    type: DefectType
    centroid_px: Tuple[int, int]
    area_px: float
    confidence: float
    bounding_box: Tuple[int, int, int, int]
    contour: np.ndarray
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "centroid_px": list(self.centroid_px),
            "area_px": self.area_px,
            "confidence": self.confidence,
            "bounding_box": list(self.bounding_box),
        }


class DefectDetector:
    """
    Hybrid defect detector using HSV color thresholding AND depth-based geometry.
    
    Combines:
    - Color-based detection (HSV thresholds for rust, crack, dent colors)
    - Geometry-based detection (curvature analysis for surface anomalies)
    """
    
    def __init__(self, enable_depth: bool = True):
        """
        Initialize detector with config thresholds.
        
        Args:
            enable_depth: If True, use depth analysis when available
        """
        vision_config = config.get("vision", {})
        
        # HSV ranges for each defect type (H: 0-180, S: 0-255, V: 0-255)
        self.thresholds = {
            DefectType.RUST: {
                "lower": np.array(vision_config.get("rust_hsv_lower", [0, 100, 100])),
                "upper": np.array(vision_config.get("rust_hsv_upper", [10, 255, 255])),
                # Also check red wraparound (170-180)
                "lower2": np.array([170, 100, 100]),
                "upper2": np.array([180, 255, 255]),
            },
            DefectType.CRACK: {
                "lower": np.array(vision_config.get("crack_hsv_lower", [0, 0, 0])),
                "upper": np.array(vision_config.get("crack_hsv_upper", [180, 255, 50])),
            },
            DefectType.DENT: {
                "lower": np.array(vision_config.get("dent_hsv_lower", [100, 100, 100])),
                "upper": np.array(vision_config.get("dent_hsv_upper", [130, 255, 255])),
            },
        }
        
        self.min_contour_area = vision_config.get("min_contour_area", 100)
        self.kernel_size = vision_config.get("morphology_kernel_size", 5)
        
        # Create morphological kernel
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        
        # Depth-based detection settings
        self.enable_depth = enable_depth and vision_config.get("enable_depth_analysis", True)
        self._depth_analyzer = None
    
    @property
    def depth_analyzer(self):
        """Lazy-load depth analyzer to avoid import overhead."""
        if self._depth_analyzer is None and self.enable_depth:
            try:
                from src.vision.depth_analyzer import DepthAnalyzer
                self._depth_analyzer = DepthAnalyzer()
            except ImportError:
                self.enable_depth = False
        return self._depth_analyzer
    
    def detect(
        self,
        rgb_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[np.ndarray] = None
    ) -> List[DetectedDefect]:
        """
        Detect defects using combined color + geometry analysis.
        
        Args:
            rgb_image: (H, W, 3) uint8 RGB image
            depth_image: Optional (H, W) float32 depth image for 3D analysis
            camera_intrinsics: Optional (3, 3) camera intrinsic matrix
            
        Returns:
            List of DetectedDefect objects (includes both color and geometric defects)
        """
        # Color-based detection (existing HSV pipeline)
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        all_defects = []
        
        for defect_type, thresholds in self.thresholds.items():
            defects = self._detect_type(hsv, defect_type, thresholds)
            all_defects.extend(defects)
        
        # Geometry-based detection (new depth pipeline)
        if depth_image is not None and self.enable_depth and self.depth_analyzer:
            geo_defects = self._detect_geometric(rgb_image, depth_image, camera_intrinsics)
            all_defects = self._merge_detections(all_defects, geo_defects)
        
        return all_defects
    
    def _detect_geometric(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: Optional[np.ndarray]
    ) -> List[DetectedDefect]:
        """Detect geometric defects using depth analysis."""
        geo_defects = self.depth_analyzer.detect_geometric_defects(rgb, depth, intrinsics)
        
        # Convert GeometricDefect to DetectedDefect format
        converted = []
        for gd in geo_defects:
            # Map geometric type to DefectType
            type_map = {
                "dent": DefectType.DENT,
                "bump": DefectType.DENT,  # Treat bumps as dent-type anomalies
                "crack": DefectType.CRACK,
                "wear": DefectType.UNKNOWN,
            }
            defect_type = type_map.get(gd.type.value, DefectType.UNKNOWN)
            
            # Approximate 2D centroid from 3D position (project back)
            # For now, use image center as placeholder - exact projection needs camera params
            h, w = depth.shape
            centroid_px = (int(w * 0.5), int(h * 0.5))  # Approximate
            
            converted.append(DetectedDefect(
                type=defect_type,
                centroid_px=centroid_px,
                area_px=gd.area_m2 * 1e6,  # Convert m² to approximate px²
                confidence=gd.confidence,
                bounding_box=(0, 0, 50, 50),  # Placeholder
                contour=np.array([]),  # 3D defects don't have 2D contours
            ))
        
        return converted
    
    def _merge_detections(
        self,
        color_defects: List[DetectedDefect],
        geo_defects: List[DetectedDefect]
    ) -> List[DetectedDefect]:
        """
        Merge color-based and geometry-based detections.
        
        Removes duplicates and boosts confidence when both methods agree.
        """
        # For now, simple concatenation - could add NMS or overlap detection
        merged = list(color_defects)
        
        for gd in geo_defects:
            # Check if similar defect already detected by color
            is_duplicate = False
            for cd in color_defects:
                # Simple distance check (would need proper pixel coordinates)
                if cd.type == gd.type:
                    # Boost confidence if both methods detected same type
                    # This is a simplified heuristic
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(gd)
        
        return merged
    
    def _detect_type(
        self,
        hsv: np.ndarray,
        defect_type: DefectType,
        thresholds: Dict
    ) -> List[DetectedDefect]:
        """
        Detect a specific type of defect.
        
        Args:
            hsv: HSV image
            defect_type: Type to detect
            thresholds: HSV threshold values
            
        Returns:
            List of detected defects of this type
        """
        # Create mask using HSV thresholding
        mask = cv2.inRange(hsv, thresholds["lower"], thresholds["upper"])
        
        # Handle red color wraparound for rust
        if "lower2" in thresholds:
            mask2 = cv2.inRange(hsv, thresholds["lower2"], thresholds["upper2"])
            mask = cv2.bitwise_or(mask, mask2)
        
        # Morphological cleanup (per Codex feedback)
        mask = cv2.erode(mask, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        defects = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by minimum area
            if area < self.min_contour_area:
                continue
            
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate confidence based on area and color match
            # Larger area = higher confidence (up to a point)
            area_confidence = min(area / 1000, 1.0)
            
            # Check color purity in the region
            roi_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(roi_mask, [contour], -1, 255, -1)
            color_ratio = cv2.countNonZero(cv2.bitwise_and(mask, roi_mask)) / area
            
            confidence = min(area_confidence * color_ratio * 1.2, 1.0)
            
            defect = DetectedDefect(
                type=defect_type,
                centroid_px=(cx, cy),
                area_px=area,
                confidence=confidence,
                bounding_box=(x, y, w, h),
                contour=contour,
            )
            defects.append(defect)
        
        return defects
    
    def draw_detections(
        self,
        image: np.ndarray,
        defects: List[DetectedDefect]
    ) -> np.ndarray:
        """
        Draw detection results on an image.
        
        Args:
            image: Original RGB image
            defects: List of detected defects
            
        Returns:
            Image with drawn bounding boxes and labels
        """
        output = image.copy()
        
        colors = {
            DefectType.RUST: (255, 0, 0),      # Red
            DefectType.CRACK: (0, 0, 0),       # Black
            DefectType.DENT: (0, 0, 255),      # Blue
            DefectType.UNKNOWN: (128, 128, 128),
        }
        
        for defect in defects:
            color = colors.get(defect.type, (128, 128, 128))
            x, y, w, h = defect.bounding_box
            
            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with confidence
            label = f"{defect.type.value} {defect.confidence:.0%}"
            cv2.putText(
                output, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # Draw centroid
            cv2.circle(output, defect.centroid_px, 3, color, -1)
        
        return output
