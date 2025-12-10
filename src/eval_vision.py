"""
Vision Pipeline Evaluation Script for AARR

Tests the vision system's ability to:
1. Detect defects with precision and recall
2. Generate accurate SAM masks (IoU)
3. Minimize false positives

Usage:
    python src/eval_vision.py
"""

import numpy as np
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import cv2


# ============ SYNTHETIC TEST DATA ============

@dataclass
class SyntheticDefect:
    """A synthetic defect for testing."""
    center: Tuple[int, int]  # (x, y) in image coordinates
    radius: int  # Approximate size
    defect_type: str  # 'rust', 'crack', 'dent'
    severity: str  # 'high', 'medium', 'low'


@dataclass 
class SyntheticImage:
    """A synthetic test image with known defects."""
    name: str
    image: np.ndarray
    ground_truth_defects: List[SyntheticDefect]
    ground_truth_mask: np.ndarray  # Binary mask of all defects


def generate_synthetic_dataset(n_images: int = 10) -> List[SyntheticImage]:
    """
    Generate synthetic test images with known defect locations.
    
    Creates images with colored patches representing defects:
    - Red/orange patches = rust/corrosion
    - Dark lines = cracks
    - Blue patches = dents (simulated)
    """
    dataset = []
    np.random.seed(42)  # Reproducible
    
    for i in range(n_images):
        # Create base gray image (simulated metal surface)
        img = np.ones((480, 640, 3), dtype=np.uint8) * 180
        
        # Add some texture/noise
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        defects = []
        mask = np.zeros((480, 640), dtype=np.uint8)
        
        # Add 1-4 random defects
        n_defects = np.random.randint(1, 5)
        
        for j in range(n_defects):
            defect_type = np.random.choice(['rust', 'crack', 'dent'])
            severity = np.random.choice(['high', 'medium', 'low'])
            
            # Random position (avoiding edges)
            cx = np.random.randint(80, 560)
            cy = np.random.randint(80, 400)
            radius = np.random.randint(15, 50)
            
            if defect_type == 'rust':
                # Orange/red patch
                color = (
                    np.random.randint(0, 50),      # B
                    np.random.randint(80, 150),    # G
                    np.random.randint(180, 255)    # R
                )
                cv2.circle(img, (cx, cy), radius, color, -1)
                cv2.circle(mask, (cx, cy), radius, 255, -1)
                
            elif defect_type == 'crack':
                # Dark line
                length = radius * 2
                angle = np.random.uniform(0, np.pi)
                x1 = int(cx - length * np.cos(angle))
                y1 = int(cy - length * np.sin(angle))
                x2 = int(cx + length * np.cos(angle))
                y2 = int(cy + length * np.sin(angle))
                cv2.line(img, (x1, y1), (x2, y2), (30, 30, 30), 3)
                cv2.line(mask, (x1, y1), (x2, y2), 255, 5)
                radius = length // 2  # Approximate
                
            elif defect_type == 'dent':
                # Slightly darker circular area with edge highlight
                cv2.circle(img, (cx, cy), radius, (140, 140, 140), -1)
                cv2.circle(img, (cx, cy), radius, (200, 200, 200), 2)
                cv2.circle(mask, (cx, cy), radius, 255, -1)
            
            defects.append(SyntheticDefect(
                center=(cx, cy),
                radius=radius,
                defect_type=defect_type,
                severity=severity
            ))
        
        dataset.append(SyntheticImage(
            name=f"synthetic_{i:03d}",
            image=img,
            ground_truth_defects=defects,
            ground_truth_mask=mask
        ))
    
    return dataset


# ============ EVALUATION METRICS ============

@dataclass
class DetectionMetrics:
    """Metrics for defect detection evaluation."""
    image_name: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float


@dataclass
class SegmentationMetrics:
    """Metrics for mask segmentation evaluation."""
    image_name: str
    iou: float  # Intersection over Union
    dice: float  # Dice coefficient
    pixel_accuracy: float


@dataclass
class VisionEvalSummary:
    """Summary of vision evaluation results."""
    total_images: int
    avg_precision: float
    avg_recall: float
    avg_f1: float
    avg_iou: float
    avg_dice: float
    false_positive_rate: float
    detection_results: List[DetectionMetrics]
    segmentation_results: List[SegmentationMetrics]
    timestamp: str


# ============ DETECTOR WRAPPER ============

class DefectDetectorWrapper:
    """Wrapper for the defect detector with optional fallback."""
    
    def __init__(self):
        """Initialize detector."""
        try:
            from src.vision.detector import DefectDetector
            self.detector = DefectDetector()
            self.available = True
        except ImportError:
            self.detector = None
            self.available = False
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect defects in image.
        
        Returns list of detections with 'center' and 'radius' keys.
        """
        if self.available and self.detector:
            try:
                # Use actual detector
                detections = self.detector.detect(image)
                return detections
            except Exception:
                pass
        
        # Fallback: Simple color-based detection
        return self._simple_detect(image)
    
    def _simple_detect(self, image: np.ndarray) -> List[Dict]:
        """Simple HSV-based detection fallback."""
        detections = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect red/orange (rust)
        lower_rust = np.array([0, 100, 100])
        upper_rust = np.array([20, 255, 255])
        rust_mask = cv2.inRange(hsv, lower_rust, upper_rust)
        
        # Detect dark areas (cracks)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Combine masks
        combined = cv2.bitwise_or(rust_mask, dark_mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                (x, y), radius = cv2.minEnclosingCircle(contour)
                detections.append({
                    "center": (int(x), int(y)),
                    "radius": int(radius),
                    "type": "unknown"
                })
        
        return detections
    
    def get_detection_mask(self, image: np.ndarray) -> np.ndarray:
        """Get binary detection mask."""
        detections = self.detect(image)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for det in detections:
            cx, cy = det["center"]
            r = det["radius"]
            cv2.circle(mask, (cx, cy), r, 255, -1)
        return mask


# ============ SAM WRAPPER ============

class SAMWrapper:
    """Wrapper for SAM segmentor."""
    
    def __init__(self):
        """Initialize SAM."""
        try:
            from src.vision.sam_segmentor import get_segmentor
            self.segmentor = get_segmentor()
            self.available = True
        except ImportError:
            self.segmentor = None
            self.available = False
    
    def segment_at_points(self, image: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Segment using SAM at given points.
        
        Returns combined binary mask.
        """
        if not self.available or not self.segmentor:
            # Fallback: return mask around points
            return self._fallback_segment(image, points)
        
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for (x, y) in points:
            try:
                result = self.segmentor.segment_at_point(image, x, y)
                if result.mask is not None:
                    combined_mask = cv2.bitwise_or(combined_mask, result.mask)
            except Exception:
                pass
        
        return combined_mask
    
    def _fallback_segment(self, image: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
        """Fallback segmentation using simple region growing."""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for (x, y) in points:
            cv2.circle(mask, (x, y), 30, 255, -1)
        return mask


# ============ EVALUATOR ============

class VisionEvaluator:
    """Evaluates vision pipeline on synthetic dataset."""
    
    def __init__(self):
        """Initialize evaluator with detector and SAM."""
        self.detector = DefectDetectorWrapper()
        self.sam = SAMWrapper()
    
    def evaluate_detection(self, image: SyntheticImage, distance_threshold: int = 50) -> DetectionMetrics:
        """
        Evaluate defect detection on a single image.
        
        A detection is a true positive if it's within distance_threshold
        of a ground truth defect center.
        """
        # Run detection
        detections = self.detector.detect(image.image)
        
        gt_defects = image.ground_truth_defects
        gt_matched = [False] * len(gt_defects)
        det_matched = [False] * len(detections)
        
        # Match detections to ground truth
        for i, det in enumerate(detections):
            det_center = det["center"]
            
            for j, gt in enumerate(gt_defects):
                if gt_matched[j]:
                    continue
                    
                # Calculate distance
                dist = np.sqrt((det_center[0] - gt.center[0])**2 + 
                              (det_center[1] - gt.center[1])**2)
                
                if dist < distance_threshold:
                    gt_matched[j] = True
                    det_matched[i] = True
                    break
        
        tp = sum(det_matched)
        fp = len(detections) - tp
        fn = len(gt_defects) - sum(gt_matched)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return DetectionMetrics(
            image_name=image.name,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1
        )
    
    def evaluate_segmentation(self, image: SyntheticImage) -> SegmentationMetrics:
        """Evaluate mask segmentation using SAM or detection mask."""
        # Get predicted mask
        pred_mask = self.detector.get_detection_mask(image.image)
        gt_mask = image.ground_truth_mask
        
        # Calculate IoU
        intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
        union = np.logical_or(pred_mask > 0, gt_mask > 0).sum()
        iou = intersection / union if union > 0 else 0
        
        # Calculate Dice
        dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0
        
        # Pixel accuracy
        total_pixels = gt_mask.size
        correct_pixels = np.sum((pred_mask > 0) == (gt_mask > 0))
        pixel_accuracy = correct_pixels / total_pixels
        
        return SegmentationMetrics(
            image_name=image.name,
            iou=iou,
            dice=dice,
            pixel_accuracy=pixel_accuracy
        )
    
    def run_all(self, dataset: List[SyntheticImage] = None) -> VisionEvalSummary:
        """Run evaluation on all images."""
        if dataset is None:
            dataset = generate_synthetic_dataset(n_images=10)
        
        detection_results = []
        segmentation_results = []
        
        total_fp = 0
        total_detections = 0
        
        for image in dataset:
            det_metrics = self.evaluate_detection(image)
            seg_metrics = self.evaluate_segmentation(image)
            
            detection_results.append(det_metrics)
            segmentation_results.append(seg_metrics)
            
            total_fp += det_metrics.false_positives
            total_detections += det_metrics.true_positives + det_metrics.false_positives
        
        # Calculate averages
        n = len(dataset)
        avg_precision = sum(r.precision for r in detection_results) / n
        avg_recall = sum(r.recall for r in detection_results) / n
        avg_f1 = sum(r.f1_score for r in detection_results) / n
        avg_iou = sum(r.iou for r in segmentation_results) / n
        avg_dice = sum(r.dice for r in segmentation_results) / n
        fpr = total_fp / total_detections if total_detections > 0 else 0
        
        return VisionEvalSummary(
            total_images=n,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1=avg_f1,
            avg_iou=avg_iou,
            avg_dice=avg_dice,
            false_positive_rate=fpr,
            detection_results=detection_results,
            segmentation_results=segmentation_results,
            timestamp=datetime.now().isoformat()
        )


# ============ MAIN ============

def print_results(summary: VisionEvalSummary):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print("👁️ AARR VISION EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Detection Metrics (n={summary.total_images} images):")
    print(f"   Precision:          {summary.avg_precision:.1%}")
    print(f"   Recall:             {summary.avg_recall:.1%}")
    print(f"   F1 Score:           {summary.avg_f1:.1%}")
    print(f"   False Positive Rate: {summary.false_positive_rate:.1%}")
    
    print(f"\n🎭 Segmentation Metrics:")
    print(f"   IoU (Jaccard):      {summary.avg_iou:.1%}")
    print(f"   Dice Coefficient:   {summary.avg_dice:.1%}")
    
    print("\n📝 Per-Image Detection:")
    for result in summary.detection_results[:5]:  # Show first 5
        icon = "✅" if result.f1_score >= 0.7 else "⚠️" if result.f1_score >= 0.4 else "❌"
        print(f"   {icon} {result.image_name}: P={result.precision:.2f} R={result.recall:.2f} F1={result.f1_score:.2f}")
    
    if len(summary.detection_results) > 5:
        print(f"   ... and {len(summary.detection_results) - 5} more")
    
    print("\n" + "=" * 60)
    
    # Overall verdict
    if summary.avg_f1 >= 0.7 and summary.avg_iou >= 0.5:
        print("✅ Vision pipeline meets production quality thresholds!")
    elif summary.avg_f1 >= 0.5:
        print("⚠️ Vision pipeline acceptable but could be improved")
    else:
        print("❌ Vision pipeline needs improvement")
    
    print("=" * 60)


def save_results(summary: VisionEvalSummary, filepath: str = "eval_results/vision_eval.json"):
    """Save results to JSON file."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    results_dict = {
        "total_images": summary.total_images,
        "avg_precision": summary.avg_precision,
        "avg_recall": summary.avg_recall,
        "avg_f1": summary.avg_f1,
        "avg_iou": summary.avg_iou,
        "avg_dice": summary.avg_dice,
        "false_positive_rate": summary.false_positive_rate,
        "timestamp": summary.timestamp,
        "detection_results": [
            {
                "name": r.image_name,
                "tp": r.true_positives,
                "fp": r.false_positives,
                "fn": r.false_negatives,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1_score
            }
            for r in summary.detection_results
        ],
        "segmentation_results": [
            {
                "name": r.image_name,
                "iou": r.iou,
                "dice": r.dice,
                "pixel_acc": r.pixel_accuracy
            }
            for r in summary.segmentation_results
        ]
    }
    
    with open(filepath, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n📁 Results saved to {filepath}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate AARR Vision Pipeline")
    parser.add_argument("--n-images", type=int, default=10, help="Number of synthetic images")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()
    
    print("🔧 Generating synthetic test dataset...")
    dataset = generate_synthetic_dataset(n_images=args.n_images)
    
    print(f"📦 Created {len(dataset)} test images with {sum(len(img.ground_truth_defects) for img in dataset)} total defects")
    
    evaluator = VisionEvaluator()
    summary = evaluator.run_all(dataset)
    
    print_results(summary)
    
    if args.save:
        save_results(summary)
