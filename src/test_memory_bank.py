"""
Tests for Memory Bank.

Tests the vector store for past repair recall.
"""

import pytest
import os
import json
import tempfile
from unittest.mock import patch


class TestRepairMemory:
    """Tests for RepairMemory dataclass."""
    
    def test_repair_memory_creation(self):
        """Test creating a RepairMemory."""
        from src.agent.memory_bank import RepairMemory
        
        memory = RepairMemory(
            id="test123",
            summary="Ground rust off turbine blade",
            timestamp="2024-01-01T00:00:00",
            part_type="turbine_blade",
            defect_type="rust",
            material="steel",
            duration_seconds=180.0,
            tool_used="grinder",
            success=True
        )
        
        assert memory.id == "test123"
        assert memory.summary == "Ground rust off turbine blade"
        assert memory.part_type == "turbine_blade"
        assert memory.duration_seconds == 180.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from src.agent.memory_bank import RepairMemory
        
        memory = RepairMemory(
            id="test123",
            summary="Test repair",
            timestamp="2024-01-01T00:00:00"
        )
        
        d = memory.to_dict()
        
        assert isinstance(d, dict)
        assert d["id"] == "test123"
        assert d["summary"] == "Test repair"
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        from src.agent.memory_bank import RepairMemory
        
        data = {
            "id": "test456",
            "summary": "Another repair",
            "timestamp": "2024-01-02T00:00:00",
            "part_type": "panel",
            "defect_type": "scratch"
        }
        
        memory = RepairMemory.from_dict(data)
        
        assert memory.id == "test456"
        assert memory.part_type == "panel"


class TestRecallResult:
    """Tests for RecallResult dataclass."""
    
    def test_recall_result_summary(self):
        """Test recall result summary generation."""
        from src.agent.memory_bank import RepairMemory, RecallResult
        
        memory = RepairMemory(
            id="test123",
            summary="Ground rust off turbine blade",
            timestamp="2024-01-01T00:00:00",
            part_type="turbine_blade",
            defect_type="rust",
            duration_seconds=180.0
        )
        
        result = RecallResult(memory=memory, similarity=0.85)
        summary = result.get_summary()
        
        assert "Past Repair" in summary
        assert "85%" in summary
        assert "turbine_blade" in summary
        assert "rust" in summary
        assert "180" in summary


class TestMemoryVectorStore:
    """Tests for MemoryVectorStore class."""
    
    def test_empty_store(self):
        """Test empty vector store."""
        from src.agent.memory_bank import MemoryVectorStore
        
        store = MemoryVectorStore()
        
        assert len(store.memories) == 0
        assert store.search("any query") == []
    
    def test_add_and_search(self):
        """Test adding memories and searching."""
        from src.agent.memory_bank import MemoryVectorStore, RepairMemory
        
        store = MemoryVectorStore()
        
        # Add a memory
        memory = RepairMemory(
            id="test1",
            summary="Ground rust off steel turbine blade using grinder",
            timestamp="2024-01-01T00:00:00",
            part_type="turbine_blade",
            defect_type="rust",
            material="steel"
        )
        store.add(memory)
        
        # Search for similar
        results = store.search("rust on metal", top_k=3)
        
        assert len(results) >= 1
        assert results[0].memory.id == "test1"
        assert results[0].similarity > 0
    
    def test_multiple_memories(self):
        """Test with multiple memories."""
        from src.agent.memory_bank import MemoryVectorStore, RepairMemory
        
        store = MemoryVectorStore()
        
        # Add memories
        store.add(RepairMemory(
            id="m1",
            summary="Polished aluminum panel with buffer",
            timestamp="2024-01-01T00:00:00",
            defect_type="scratch",
            material="aluminum"
        ))
        
        store.add(RepairMemory(
            id="m2",
            summary="Ground rust off steel part",
            timestamp="2024-01-02T00:00:00",
            defect_type="rust",
            material="steel"
        ))
        
        # Search for rust
        results = store.search("rust on steel", top_k=3)
        
        # Should find rust memory first
        assert len(results) >= 1
        assert results[0].memory.defect_type == "rust"
    
    def test_persistence(self):
        """Test saving and loading memories."""
        from src.agent.memory_bank import MemoryVectorStore, RepairMemory
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Create store and add memory
            store1 = MemoryVectorStore(persistence_path=temp_path)
            store1.add(RepairMemory(
                id="persist1",
                summary="Test persistence memory",
                timestamp="2024-01-01T00:00:00"
            ))
            
            # Create new store from same file
            store2 = MemoryVectorStore(persistence_path=temp_path)
            
            assert len(store2.memories) == 1
            assert store2.memories[0].id == "persist1"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_clear(self):
        """Test clearing memories."""
        from src.agent.memory_bank import MemoryVectorStore, RepairMemory
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            store = MemoryVectorStore(persistence_path=temp_path)
            store.add(RepairMemory(
                id="clear1",
                summary="Memory to clear",
                timestamp="2024-01-01T00:00:00"
            ))
            
            assert len(store.memories) == 1
            
            store.clear()
            
            assert len(store.memories) == 0
            assert not os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_get_stats(self):
        """Test statistics generation."""
        from src.agent.memory_bank import MemoryVectorStore, RepairMemory
        
        store = MemoryVectorStore()
        
        # Empty stats
        stats = store.get_stats()
        assert stats["total_memories"] == 0
        
        # Add memories
        store.add(RepairMemory(
            id="s1", summary="Rust repair",
            timestamp="2024-01-01T00:00:00",
            defect_type="rust", part_type="blade"
        ))
        store.add(RepairMemory(
            id="s2", summary="Another rust repair",
            timestamp="2024-01-02T00:00:00",
            defect_type="rust", part_type="panel"
        ))
        
        stats = store.get_stats()
        
        assert stats["total_memories"] == 2
        assert stats["defect_types"]["rust"] == 2
        assert "blade" in stats["part_types"]


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_store_repair(self):
        """Test storing a repair."""
        from src.agent.memory_bank import store_repair, get_memory_store, clear_all_memories
        
        # Clear first
        clear_all_memories()
        
        memory = store_repair(
            summary="Test store function",
            part_type="test_part",
            defect_type="test_defect"
        )
        
        assert memory.summary == "Test store function"
        assert len(memory.id) > 0
        
        # Verify it's in the store
        store = get_memory_store()
        assert len(store.memories) >= 1
        
        # Clean up
        clear_all_memories()
    
    def test_recall_similar_repairs(self):
        """Test recalling similar repairs."""
        from src.agent.memory_bank import (
            store_repair, recall_similar_repairs, clear_all_memories
        )
        
        # Clear and add test data
        clear_all_memories()
        
        store_repair(
            summary="Repaired rust on steel blade",
            defect_type="rust",
            material="steel"
        )
        
        results = recall_similar_repairs("rust steel")
        
        assert len(results) >= 1
        assert "rust" in results[0].memory.summary.lower()
        
        # Clean up
        clear_all_memories()
    
    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        from src.agent.memory_bank import get_memory_stats, clear_all_memories
        
        clear_all_memories()
        stats = get_memory_stats()
        
        assert "total_memories" in stats
        assert stats["total_memories"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
