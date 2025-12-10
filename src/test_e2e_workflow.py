"""
End-to-End Workflow Tests for AARR

Tests the complete workflow pipeline:
1. Mesh loading and visualization
2. Scan → Detect → Plan → Execute flow
3. Agent chat integration
4. Performance benchmarks

Usage:
    pytest src/test_e2e_workflow.py -v
"""

import pytest
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path


# ============ FIXTURES ============

@pytest.fixture
def premium_meshes_dir():
    """Get the premium meshes directory."""
    return Path("assets/premium_meshes")


@pytest.fixture
def sample_mesh_path(premium_meshes_dir):
    """Get a sample mesh path, generating if needed."""
    mesh_path = premium_meshes_dir / "turbine_blade.stl"
    if not mesh_path.exists():
        from src.visualization.premium_meshes import generate_premium_meshes
        generate_premium_meshes(str(premium_meshes_dir))
    return mesh_path


@pytest.fixture
def sample_defects():
    """Sample defects for testing."""
    return [
        {
            "position": (0.1, 0.2, 0.15),
            "type": "crack",
            "severity": "high",
            "confidence": 0.92,
            "normal": (0, 0, 1)
        },
        {
            "position": (0.3, 0.1, 0.18),
            "type": "rust",
            "severity": "medium",
            "confidence": 0.85,
            "normal": (0, 0, 1)
        },
        {
            "position": (0.2, 0.3, 0.12),
            "type": "dent",
            "severity": "low",
            "confidence": 0.78,
            "normal": (0, 1, 0)
        },
    ]


@pytest.fixture
def agent_team():
    """Initialize the agent team."""
    from src.agent.supervisor_agent import ConversationalTeam
    return ConversationalTeam()


# ============ MESH LOADING TESTS ============

class TestMeshLoading:
    """Tests for mesh loading and visualization."""
    
    def test_load_stl_mesh(self, sample_mesh_path):
        """Test that STL mesh loads successfully."""
        from src.visualization.mesh_loader import load_mesh
        
        mesh_data = load_mesh(str(sample_mesh_path))
        
        assert mesh_data is not None
        assert mesh_data.vertices is not None
        assert mesh_data.faces is not None
        assert len(mesh_data.vertices) > 0
        assert len(mesh_data.faces) > 0
    
    def test_mesh_data_attributes(self, sample_mesh_path):
        """Test MeshData has expected attributes."""
        from src.visualization.mesh_loader import load_mesh
        
        mesh_data = load_mesh(str(sample_mesh_path))
        
        assert hasattr(mesh_data, "vertices")
        assert hasattr(mesh_data, "faces")
        assert hasattr(mesh_data, "name")
        assert mesh_data.name == "turbine_blade"
    
    def test_sample_surface_points(self, sample_mesh_path):
        """Test sampling points from mesh surface."""
        from src.visualization.mesh_loader import load_mesh, sample_surface_points
        
        mesh_data = load_mesh(str(sample_mesh_path))
        positions, normals = sample_surface_points(mesh_data, n_points=5)
        
        assert len(positions) == 5
        assert len(normals) == 5
        assert all(len(p) == 3 for p in positions)
    
    def test_procedural_part_generation(self):
        """Test procedural part generators work."""
        from src.visualization.premium_procedural_models import (
            generate_premium_pipe,
            generate_turbine_blade_v2,
        )
        
        pipe_trace = generate_premium_pipe()
        blade_trace = generate_turbine_blade_v2()
        
        assert pipe_trace is not None
        assert blade_trace is not None
    
    def test_premium_defects(self):
        """Test premium defect metadata is available."""
        from src.visualization.premium_meshes import get_premium_defects
        
        defects = get_premium_defects("turbine_blade")
        
        assert defects is not None
        assert len(defects) > 0
        assert all("position" in d for d in defects)
        assert all("type" in d for d in defects)


# ============ WORKFLOW PIPELINE TESTS ============

class TestWorkflowPipeline:
    """Tests for the scan-detect-plan-execute workflow."""
    
    def test_defect_detection_workflow(self, sample_mesh_path):
        """Test the defect detection part of the workflow."""
        from src.visualization.mesh_loader import load_mesh, sample_surface_points
        
        # Step 1: Load mesh
        mesh_data = load_mesh(str(sample_mesh_path))
        assert mesh_data is not None
        
        # Step 2: Sample points (simulates scan)
        positions, normals = sample_surface_points(mesh_data, n_points=3)
        assert len(positions) == 3
        
        # Step 3: Create synthetic defects (simulates detection)
        defects = [
            {
                "position": tuple(positions[i]),
                "type": np.random.choice(["crack", "rust", "dent"]),
                "severity": np.random.choice(["high", "medium", "low"]),
                "confidence": np.random.uniform(0.7, 0.95),
                "normal": tuple(normals[i]) if i < len(normals) else (0, 0, 1)
            }
            for i in range(len(positions))
        ]
        
        assert len(defects) == 3
        assert all("position" in d for d in defects)
        assert all("severity" in d for d in defects)
    
    def test_plan_generation(self, sample_defects):
        """Test repair plan generation."""
        from src.agent.tools import get_fallback_plan
        
        plans = []
        for i, defect in enumerate(sample_defects):
            plan = get_fallback_plan(defect["type"])
            plans.append({"index": i, "defect_type": defect["type"], **plan})
        
        assert len(plans) == len(sample_defects)
        assert all("tool" in p for p in plans)
        # Check for strategy and estimated_time_seconds (actual keys in fallback plans)
        assert all("strategy" in p and "estimated_time_seconds" in p for p in plans)
    
    def test_path_optimization(self, sample_defects):
        """Test TSP path optimization."""
        from src.planning.tsp import optimize_with_metrics
        
        result = optimize_with_metrics(
            sample_defects,
            robot_pos=(0, 0, 0.5)
        )
        
        assert result is not None
        assert len(result.optimized_order) == len(sample_defects)
        assert result.efficiency_gain_percent >= 0
        # Algorithm name includes both phases
        assert "nearest" in result.algorithm_used.lower() or "2-opt" in result.algorithm_used.lower()
    
    def test_full_pipeline_timing(self, sample_mesh_path, sample_defects):
        """Test full pipeline executes within time budget."""
        from src.visualization.mesh_loader import load_mesh
        from src.agent.tools import get_fallback_plan
        from src.planning.tsp import optimize_with_metrics
        
        start = time.time()
        
        # Load mesh
        mesh_data = load_mesh(str(sample_mesh_path))
        
        # Generate plans
        plans = [get_fallback_plan(d["type"]) for d in sample_defects]
        
        # Optimize path
        result = optimize_with_metrics(sample_defects, robot_pos=(0, 0, 0.5))
        
        elapsed = time.time() - start
        
        # Should complete within 2 seconds
        assert elapsed < 2.0, f"Pipeline took {elapsed:.2f}s, expected < 2s"


# ============ AGENT INTEGRATION TESTS ============

class TestAgentIntegration:
    """Tests for agent chat integration."""
    
    def test_agent_team_initialization(self, agent_team):
        """Test ConversationalTeam initializes properly."""
        assert agent_team is not None
        # ConversationalTeam uses 'process_message' method
        assert hasattr(agent_team, "process_message") or hasattr(agent_team, "chat") or callable(agent_team)
    
    def test_knowledge_base_consult(self):
        """Test RAG knowledge base consultation."""
        from src.agent.knowledge_base import consult_manual
        
        result = consult_manual("How do we fix Steel?")
        
        assert result is not None
        assert "SOP" in result or "Steel" in result or "Grinder" in result
    
    def test_ml_predictor_integration(self, sample_defects):
        """Test ML predictor returns valid predictions."""
        from src.ml import get_predictor
        
        predictor = get_predictor()
        
        for defect in sample_defects:
            result = predictor.predict_for_defect(defect)
            
            # Result is a PredictionResult dataclass
            assert result is not None
            assert hasattr(result, 'repair_time_seconds')
            assert result.repair_time_seconds > 0


# ============ PERFORMANCE BENCHMARKS ============

class TestPerformanceBenchmarks:
    """Performance benchmarks for the system."""
    
    def test_mesh_load_latency(self, sample_mesh_path):
        """Benchmark mesh loading time."""
        from src.visualization.mesh_loader import load_mesh
        
        times = []
        for _ in range(5):
            start = time.time()
            load_mesh(str(sample_mesh_path))
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        
        # Should load within 500ms
        assert avg_time < 0.5, f"Mesh load avg: {avg_time*1000:.1f}ms"
        print(f"\n📊 Mesh load avg: {avg_time*1000:.1f}ms")
    
    def test_tsp_optimization_latency(self, sample_defects):
        """Benchmark TSP optimization time."""
        from src.planning.tsp import optimize_with_metrics
        
        # Add more defects to stress test
        many_defects = sample_defects * 10  # 30 defects
        
        start = time.time()
        result = optimize_with_metrics(many_defects, robot_pos=(0, 0, 0.5))
        elapsed = time.time() - start
        
        # Should complete within 1 second even for 30 defects
        assert elapsed < 1.0, f"TSP took {elapsed*1000:.1f}ms"
        print(f"\n📊 TSP optimization (n={len(many_defects)}): {elapsed*1000:.1f}ms")
    
    def test_ml_prediction_latency(self, sample_defects):
        """Benchmark ML prediction time."""
        from src.ml import get_predictor
        
        predictor = get_predictor()
        
        times = []
        for defect in sample_defects:
            start = time.time()
            predictor.predict_for_defect(defect)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        
        # Should predict within 100ms per defect (includes model load overhead)
        assert avg_time < 0.1, f"ML predict avg: {avg_time*1000:.1f}ms"
        print(f"\n📊 ML prediction avg: {avg_time*1000:.1f}ms")
    
    def test_knowledge_base_latency(self):
        """Benchmark RAG knowledge base query time."""
        from src.agent.knowledge_base import consult_manual
        
        queries = [
            "How do we fix Steel?",
            "Aluminum repair settings",
            "Rust treatment procedure"
        ]
        
        times = []
        for query in queries:
            start = time.time()
            consult_manual(query)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        
        # Should query within 100ms
        assert avg_time < 0.1, f"RAG query avg: {avg_time*1000:.1f}ms"
        print(f"\n📊 RAG query avg: {avg_time*1000:.1f}ms")


# ============ CODE INTERPRETER SECURITY ============

class TestCodeInterpreterSecurity:
    """Security tests for the code interpreter sandbox."""
    
    def test_safe_code_executes(self):
        """Test safe code executes successfully."""
        from src.planning.code_interpreter import exec_custom_path
        
        safe_code = """
import numpy as np
import math

def generate_custom_path(center, radius):
    points = []
    for i in range(8):
        angle = i * 2 * math.pi / 8
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        z = center[2]
        points.append([x, y, z])
    return np.array(points)
"""
        
        result = exec_custom_path(safe_code)
        
        assert result.success
        assert result.points is not None
        assert result.points.shape == (8, 3)
    
    def test_os_import_blocked(self):
        """Test os import is blocked."""
        from src.planning.code_interpreter import validate_generated_code
        
        malicious_code = """
import os

def generate_custom_path(center, radius):
    os.system("rm -rf /")
    return np.array([[0, 0, 0]])
"""
        
        is_safe, error = validate_generated_code(malicious_code)
        
        assert not is_safe
        assert "os" in error.lower() or "forbidden" in error.lower()
    
    def test_subprocess_blocked(self):
        """Test subprocess is blocked."""
        from src.planning.code_interpreter import validate_generated_code
        
        malicious_code = """
import subprocess

def generate_custom_path(center, radius):
    subprocess.run(["ls", "-la"])
    return np.array([[0, 0, 0]])
"""
        
        is_safe, error = validate_generated_code(malicious_code)
        
        assert not is_safe


# ============ MAIN ============

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
