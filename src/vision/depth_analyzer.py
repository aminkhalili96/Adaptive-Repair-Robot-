"""
RGBD Depth Analyzer - 3D Point Cloud Based Defect Detection.

This module converts RGB + Depth images into point clouds and performs
geometric analysis to detect defects that color-based methods might miss:
- Dents (concave regions)
- Bumps (convex regions)
- Surface irregularities (high curvature areas)
- Cracks (depth discontinuities)

Architecture:
    RGB + Depth → Point Cloud (XYZRGB)
    Point Cloud → Normal Estimation (PCA-based)
    Normals → Curvature Computation
    High Curvature → DBSCAN Clustering → Defect Regions
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Try to import open3d for efficient point cloud processing
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class GeometricDefectType(Enum):
    """Types of geometric defects detectable via depth analysis."""
    DENT = "dent"           # Concave region (negative curvature)
    BUMP = "bump"           # Convex protrusion (positive curvature)
    CRACK = "crack"         # Depth discontinuity
    SURFACE_WEAR = "wear"   # Irregular surface texture
    UNKNOWN = "unknown"


@dataclass
class GeometricDefect:
    """
    A geometric defect detected from depth analysis.
    
    Attributes:
        type: Type of geometric defect
        position_3d: [x, y, z] world coordinates of defect centroid
        normal: [nx, ny, nz] estimated surface normal
        area_m2: Approximate area in square meters
        depth_deviation: How much depth deviates from expected surface
        curvature: Mean curvature value at defect region
        confidence: Detection confidence (0-1)
        point_indices: Indices of points belonging to this defect
    """
    type: GeometricDefectType
    position_3d: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    area_m2: float
    depth_deviation: float
    curvature: float
    confidence: float
    point_indices: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "position_3d": list(self.position_3d),
            "normal": list(self.normal),
            "area_m2": float(self.area_m2),
            "depth_deviation": float(self.depth_deviation),
            "curvature": float(self.curvature),
            "confidence": float(self.confidence),
            "num_points": len(self.point_indices),
        }


class DepthAnalyzer:
    """
    Analyzes RGBD images for geometric defects using point cloud processing.
    
    This class provides true 3D defect detection by:
    1. Converting depth images to point clouds
    2. Estimating surface normals
    3. Computing local curvature
    4. Clustering high-curvature regions as defects
    
    Example:
        >>> analyzer = DepthAnalyzer()
        >>> defects = analyzer.detect_geometric_defects(rgb, depth, camera_intrinsics)
        >>> for d in defects:
        ...     print(f"{d.type.value} at {d.position_3d}, curvature={d.curvature:.3f}")
    """
    
    def __init__(
        self,
        voxel_size: float = 0.005,
        normal_k_neighbors: int = 20,
        curvature_threshold: float = 0.03,
        clustering_eps: float = 0.015,
        min_defect_points: int = 30,
        max_depth: float = 2.0,
    ):
        """
        Initialize the depth analyzer.
        
        Args:
            voxel_size: Voxel size for downsampling (meters). Smaller = more detail.
            normal_k_neighbors: Number of neighbors for normal estimation.
            curvature_threshold: Curvature above this value indicates defect.
            clustering_eps: DBSCAN epsilon for clustering defect points.
            min_defect_points: Minimum points to constitute a defect.
            max_depth: Maximum valid depth value (meters).
        """
        self.voxel_size = voxel_size
        self.normal_k_neighbors = normal_k_neighbors
        self.curvature_threshold = curvature_threshold
        self.clustering_eps = clustering_eps
        self.min_defect_points = min_defect_points
        self.max_depth = max_depth
        
        self._use_open3d = HAS_OPEN3D
        
        if not HAS_OPEN3D:
            logger.warning("open3d not available - using numpy fallback (slower)")
    
    def depth_to_pointcloud(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        max_depth: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert RGB + Depth images to an XYZRGB point cloud.
        
        Args:
            rgb: (H, W, 3) uint8 RGB image
            depth: (H, W) float32 depth image (meters or normalized)
            intrinsics: (3, 3) camera intrinsic matrix K
            max_depth: Maximum valid depth (default: self.max_depth)
            
        Returns:
            (N, 6) array of [X, Y, Z, R, G, B] points in camera frame
        """
        max_depth = max_depth or self.max_depth
        h, w = depth.shape
        
        # Get camera parameters from intrinsics
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Create pixel coordinate grids
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)
        
        # Filter valid depth values
        # PyBullet depth buffer needs conversion: linearize if needed
        valid_depth = self._linearize_depth(depth, max_depth)
        valid_mask = (valid_depth > 0.01) & (valid_depth < max_depth)
        
        # Compute 3D coordinates
        z = valid_depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack into point cloud
        points = np.stack([x, y, z], axis=-1)  # (H, W, 3)
        colors = rgb.astype(np.float32) / 255.0  # Normalize colors
        
        # Flatten and filter
        points_flat = points.reshape(-1, 3)
        colors_flat = colors.reshape(-1, 3)
        valid_flat = valid_mask.reshape(-1)
        
        valid_points = points_flat[valid_flat]
        valid_colors = colors_flat[valid_flat]
        
        # Combine XYZRGB
        pointcloud = np.hstack([valid_points, valid_colors])
        
        return pointcloud
    
    def _linearize_depth(self, depth: np.ndarray, max_depth: float) -> np.ndarray:
        """
        Convert PyBullet depth buffer to linear depth in meters.
        
        PyBullet uses a non-linear depth buffer. This function converts it
        to actual distances.
        """
        # Check if already linearized (values in reasonable range)
        if depth.max() < 10:
            return depth
        
        # PyBullet depth buffer linearization
        # depth_linear = far * near / (far - (far - near) * depth_buffer)
        near = 0.1
        far = max_depth
        
        # Avoid division by zero
        depth_buffer = np.clip(depth, 0.0001, 0.9999)
        linear_depth = far * near / (far - (far - near) * depth_buffer)
        
        return linear_depth
    
    def estimate_normals(
        self,
        points: np.ndarray,
        k_neighbors: Optional[int] = None
    ) -> np.ndarray:
        """
        Estimate surface normals using PCA on local neighborhoods.
        
        Args:
            points: (N, 3) or (N, 6) point cloud (uses XYZ only)
            k_neighbors: Number of neighbors for PCA
            
        Returns:
            (N, 3) array of unit normal vectors
        """
        k = k_neighbors or self.normal_k_neighbors
        xyz = points[:, :3] if points.shape[1] > 3 else points
        n_points = len(xyz)
        
        if n_points < k:
            logger.warning(f"Too few points ({n_points}) for normal estimation")
            return np.tile([0, 0, 1], (n_points, 1))
        
        if self._use_open3d:
            return self._estimate_normals_o3d(xyz, k)
        else:
            return self._estimate_normals_numpy(xyz, k)
    
    def _estimate_normals_o3d(self, xyz: np.ndarray, k: int) -> np.ndarray:
        """Estimate normals using Open3D (fast)."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
        )
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
        return np.asarray(pcd.normals)
    
    def _estimate_normals_numpy(self, xyz: np.ndarray, k: int) -> np.ndarray:
        """Estimate normals using numpy (slower but no dependencies)."""
        n_points = len(xyz)
        normals = np.zeros((n_points, 3))
        
        # Build k-NN
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(xyz)
        _, indices = nbrs.kneighbors(xyz)
        
        # PCA for each point
        for i in range(n_points):
            neighbors = xyz[indices[i]]
            centered = neighbors - neighbors.mean(axis=0)
            
            # Covariance matrix
            cov = np.dot(centered.T, centered) / k
            
            # Smallest eigenvector is the normal
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normals[i] = eigenvectors[:, 0]  # Smallest eigenvalue
            
            # Orient towards camera (assume camera at origin)
            if np.dot(normals[i], -xyz[i]) < 0:
                normals[i] = -normals[i]
        
        return normals
    
    def compute_curvature(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        k_neighbors: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute local curvature as the variation of normals.
        
        High curvature indicates surface features like dents, bumps, or edges.
        
        Args:
            points: (N, 3+) point cloud
            normals: (N, 3) surface normals
            k_neighbors: Neighbors to consider
            
        Returns:
            (N,) array of curvature values (0 = flat, high = curved)
        """
        k = k_neighbors or self.normal_k_neighbors
        xyz = points[:, :3] if points.shape[1] > 3 else points
        n_points = len(xyz)
        
        if n_points < k:
            return np.zeros(n_points)
        
        # Build k-NN
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(xyz)
        _, indices = nbrs.kneighbors(xyz)
        
        curvatures = np.zeros(n_points)
        
        for i in range(n_points):
            neighbor_normals = normals[indices[i]]
            
            # Curvature = 1 - consistency of normals
            # Low variance = flat surface, high variance = curved
            normal_mean = neighbor_normals.mean(axis=0)
            normal_mean /= np.linalg.norm(normal_mean) + 1e-8
            
            # Dot product with mean normal
            dots = np.abs(np.dot(neighbor_normals, normal_mean))
            curvatures[i] = 1.0 - dots.mean()
        
        return curvatures
    
    def detect_geometric_defects(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: Optional[np.ndarray] = None
    ) -> List[GeometricDefect]:
        """
        Detect geometric defects in an RGBD image.
        
        This is the main entry point for 3D defect detection.
        
        Args:
            rgb: (H, W, 3) uint8 RGB image
            depth: (H, W) float32 depth image
            intrinsics: (3, 3) camera intrinsic matrix (uses default if None)
            
        Returns:
            List of GeometricDefect objects
        """
        # Default intrinsics if not provided
        if intrinsics is None:
            h, w = depth.shape
            fx = fy = w / (2 * np.tan(np.radians(30)))  # Assume 60° FOV
            intrinsics = np.array([
                [fx, 0, w/2],
                [0, fy, h/2],
                [0, 0, 1]
            ])
        
        # Step 1: Convert to point cloud
        pointcloud = self.depth_to_pointcloud(rgb, depth, intrinsics)
        
        if len(pointcloud) < self.min_defect_points:
            logger.warning(f"Too few valid points: {len(pointcloud)}")
            return []
        
        # Step 2: Downsample for efficiency
        if self._use_open3d:
            pointcloud = self._downsample_o3d(pointcloud)
        
        # Step 3: Estimate normals
        normals = self.estimate_normals(pointcloud)
        
        # Step 4: Compute curvature
        curvatures = self.compute_curvature(pointcloud, normals)
        
        # Step 5: Find high-curvature regions
        high_curvature_mask = curvatures > self.curvature_threshold
        high_curvature_indices = np.where(high_curvature_mask)[0]
        
        if len(high_curvature_indices) < self.min_defect_points:
            return []
        
        # Step 6: Cluster defect points using DBSCAN
        defect_points = pointcloud[high_curvature_indices, :3]
        clustering = DBSCAN(
            eps=self.clustering_eps,
            min_samples=max(3, self.min_defect_points // 5)
        ).fit(defect_points)
        
        labels = clustering.labels_
        unique_labels = set(labels) - {-1}  # Exclude noise
        
        # Step 7: Build defect objects
        defects = []
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_indices = high_curvature_indices[cluster_mask]
            
            if len(cluster_indices) < self.min_defect_points:
                continue
            
            defect = self._build_defect(
                pointcloud, normals, curvatures, cluster_indices
            )
            if defect:
                defects.append(defect)
        
        # Sort by confidence
        defects.sort(key=lambda d: d.confidence, reverse=True)
        
        return defects
    
    def _downsample_o3d(self, pointcloud: np.ndarray) -> np.ndarray:
        """Downsample point cloud using Open3D voxel grid."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
        if pointcloud.shape[1] >= 6:
            pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:6])
        
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        xyz = np.asarray(pcd_down.points)
        if pcd_down.has_colors():
            colors = np.asarray(pcd_down.colors)
            return np.hstack([xyz, colors])
        return xyz
    
    def _build_defect(
        self,
        pointcloud: np.ndarray,
        normals: np.ndarray,
        curvatures: np.ndarray,
        indices: np.ndarray
    ) -> Optional[GeometricDefect]:
        """Build a GeometricDefect object from clustered points."""
        points = pointcloud[indices, :3]
        point_normals = normals[indices]
        point_curvatures = curvatures[indices]
        
        # Compute centroid
        centroid = points.mean(axis=0)
        
        # Average normal
        avg_normal = point_normals.mean(axis=0)
        avg_normal /= np.linalg.norm(avg_normal) + 1e-8
        
        # Mean curvature
        mean_curvature = point_curvatures.mean()
        
        # Estimate area (convex hull approximation)
        try:
            from scipy.spatial import ConvexHull
            if len(points) >= 4:
                hull = ConvexHull(points[:, :2])  # 2D projection
                area = hull.volume  # In 2D, volume = area
            else:
                area = 0.001
        except Exception:
            area = len(points) * (self.voxel_size ** 2)
        
        # Estimate depth deviation from local plane
        centered = points - centroid
        if len(centered) > 3:
            _, s, _ = np.linalg.svd(centered)
            depth_deviation = s[-1] if len(s) > 0 else 0
        else:
            depth_deviation = 0
        
        # Classify defect type based on geometry
        defect_type = self._classify_defect_type(
            avg_normal, mean_curvature, depth_deviation, points
        )
        
        # Confidence based on curvature strength and cluster size
        size_factor = min(1.0, len(indices) / 200)
        curvature_factor = min(1.0, mean_curvature / (self.curvature_threshold * 3))
        confidence = 0.6 * curvature_factor + 0.4 * size_factor
        
        return GeometricDefect(
            type=defect_type,
            position_3d=tuple(centroid),
            normal=tuple(avg_normal),
            area_m2=float(area),
            depth_deviation=float(depth_deviation),
            curvature=float(mean_curvature),
            confidence=float(min(1.0, confidence)),
            point_indices=indices,
        )
    
    def _classify_defect_type(
        self,
        normal: np.ndarray,
        curvature: float,
        depth_deviation: float,
        points: np.ndarray
    ) -> GeometricDefectType:
        """Classify defect type based on geometric features."""
        # Check if concave (dent) or convex (bump)
        # by comparing point positions to expected surface
        centroid = points.mean(axis=0)
        
        # Use the normal direction to determine in/out
        # Points are a dent if they're "behind" the expected surface
        point_to_centroid = points - centroid
        projections = np.dot(point_to_centroid, normal)
        
        if projections.std() > depth_deviation * 2:
            # High depth variation = possible crack
            return GeometricDefectType.CRACK
        
        if depth_deviation > self.voxel_size * 5:
            # Significant depth deviation
            if projections.mean() < 0:
                return GeometricDefectType.DENT
            else:
                return GeometricDefectType.BUMP
        
        if curvature > self.curvature_threshold * 2:
            return GeometricDefectType.SURFACE_WEAR
        
        return GeometricDefectType.UNKNOWN
    
    def get_status(self) -> Dict[str, Any]:
        """Get analyzer configuration status."""
        return {
            "open3d_available": HAS_OPEN3D,
            "using_open3d": self._use_open3d,
            "voxel_size": self.voxel_size,
            "curvature_threshold": self.curvature_threshold,
            "min_defect_points": self.min_defect_points,
        }


# ============ CONVENIENCE FUNCTIONS ============

def analyze_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: Optional[np.ndarray] = None
) -> List[Dict]:
    """
    Quick function to analyze RGBD for defects.
    
    Returns list of defect dictionaries.
    """
    analyzer = DepthAnalyzer()
    defects = analyzer.detect_geometric_defects(rgb, depth, intrinsics)
    return [d.to_dict() for d in defects]


def depth_to_pointcloud_quick(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray
) -> np.ndarray:
    """Quick function to convert RGBD to point cloud."""
    analyzer = DepthAnalyzer()
    return analyzer.depth_to_pointcloud(rgb, depth, intrinsics)
