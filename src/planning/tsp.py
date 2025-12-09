"""
TSP (Traveling Salesman Problem) solver for multi-defect ordering.

Uses nearest-neighbor heuristic with optional 2-opt improvement
per Codex feedback.
"""

import numpy as np
from typing import List, TypeVar
from dataclasses import dataclass

T = TypeVar('T')


def distance(pos1: tuple, pos2: tuple) -> float:
    """
    Calculate Euclidean distance between two 3D positions.
    
    Args:
        pos1: (x, y, z) first position
        pos2: (x, y, z) second position
        
    Returns:
        Distance between positions
    """
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))


def nearest_neighbor_tsp(
    items: List[T],
    get_position: callable,
    start_pos: tuple = None
) -> List[T]:
    """
    Order items using nearest-neighbor heuristic.
    
    Args:
        items: List of items to order
        get_position: Function that extracts (x, y, z) position from an item
        start_pos: Optional starting position (default: first item)
        
    Returns:
        Reordered list minimizing total travel distance
    """
    if len(items) <= 1:
        return items
    
    remaining = list(items)
    ordered = []
    
    # Start from first item or given position
    if start_pos is None:
        current = remaining.pop(0)
        ordered.append(current)
        current_pos = get_position(current)
    else:
        current_pos = start_pos
    
    while remaining:
        # Find nearest
        nearest_idx = min(
            range(len(remaining)),
            key=lambda i: distance(current_pos, get_position(remaining[i]))
        )
        
        current = remaining.pop(nearest_idx)
        ordered.append(current)
        current_pos = get_position(current)
    
    return ordered


def two_opt_improve(
    items: List[T],
    get_position: callable,
    max_iterations: int = 100
) -> List[T]:
    """
    Improve TSP solution using 2-opt swaps (per Codex feedback).
    
    Args:
        items: Initial ordering
        get_position: Function to extract position
        max_iterations: Maximum improvement iterations
        
    Returns:
        Improved ordering
    """
    if len(items) <= 2:
        return items
    
    def total_distance(order):
        return sum(
            distance(get_position(order[i]), get_position(order[i + 1]))
            for i in range(len(order) - 1)
        )
    
    best = list(items)
    best_dist = total_distance(best)
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(len(best) - 2):
            for j in range(i + 2, len(best)):
                # Try reversing segment between i+1 and j
                new_order = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                new_dist = total_distance(new_order)
                
                if new_dist < best_dist:
                    best = new_order
                    best_dist = new_dist
                    improved = True
                    break
            
            if improved:
                break
    
    return best


def optimize_defect_order(
    defects: List,
    robot_pos: tuple = (0, 0, 0.5),
    use_2opt: bool = True
) -> List:
    """
    Optimize the order of defects to minimize robot travel.
    
    Args:
        defects: List of defects (with .position attribute or Pose3D objects)
        robot_pos: Starting robot position
        use_2opt: Whether to apply 2-opt improvement
        
    Returns:
        Reordered defect list
    """
    if len(defects) <= 1:
        return defects
    
    # Extract position from defect
    def get_pos(defect):
        if hasattr(defect, 'position'):
            return defect.position
        elif hasattr(defect, 'centroid_px'):
            # For DetectedDefect, use centroid (will be 2D but still useful for ordering)
            return (*defect.centroid_px, 0)
        else:
            return (0, 0, 0)
    
    # Initial ordering with nearest neighbor
    ordered = nearest_neighbor_tsp(defects, get_pos, robot_pos)
    
    # Improve with 2-opt if requested
    if use_2opt:
        ordered = two_opt_improve(ordered, get_pos)
    
    return ordered
