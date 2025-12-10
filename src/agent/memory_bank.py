"""
Memory Bank - Agentic Memory for Past Repairs.

Stores and retrieves past repair experiences using vector embeddings,
enabling the agent to learn from history and provide contextual recommendations.

Usage:
    from src.agent.memory_bank import store_repair, recall_similar_repairs
    
    # Store a completed repair
    store_repair(summary="Repaired rust on turbine blade using grinding", 
                 metadata={"part": "turbine", "defect": "rust", "duration": 120})
    
    # Recall similar repairs
    results = recall_similar_repairs("How to fix rust on metal?")
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np


# ============ DATA STRUCTURES ============

@dataclass
class RepairMemory:
    """A single stored repair experience."""
    id: str
    summary: str
    timestamp: str
    part_type: str = ""
    defect_type: str = ""
    material: str = ""
    duration_seconds: float = 0.0
    tool_used: str = ""
    success: bool = True
    notes: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RepairMemory':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RecallResult:
    """Result from memory recall."""
    memory: RepairMemory
    similarity: float
    
    def get_summary(self) -> str:
        """Get formatted summary for display."""
        return (
            f"**Past Repair** (Match: {self.similarity:.0%})\n"
            f"- Part: {self.memory.part_type or 'Unknown'}\n"
            f"- Defect: {self.memory.defect_type or 'Unknown'}\n"
            f"- Action: {self.memory.summary}\n"
            f"- Duration: {self.memory.duration_seconds:.0f}s\n"
            f"- Date: {self.memory.timestamp[:10]}"
        )


# ============ VECTOR STORE ============

class MemoryVectorStore:
    """
    Simple vector store for repair memories.
    
    Uses TF-IDF-like word vectors with cosine similarity.
    For production, consider FAISS or a vector database.
    """
    
    def __init__(self, persistence_path: Optional[str] = None):
        self.memories: List[RepairMemory] = []
        self.vectors: Optional[np.ndarray] = None
        self.vocabulary: List[str] = []
        self.word_to_idx: Dict[str, int] = {}
        self.persistence_path = persistence_path
        
        # Load persisted memories if available
        if persistence_path and os.path.exists(persistence_path):
            self._load()
    
    def _tokenize(self, text: str) -> set:
        """Simple word tokenization."""
        import re
        words = re.findall(r'[a-zA-Z]+', text.lower())
        return set(words)
    
    def _build_vocabulary(self):
        """Build vocabulary from all memories."""
        all_words = set()
        for mem in self.memories:
            # Combine all text fields
            text = f"{mem.summary} {mem.part_type} {mem.defect_type} {mem.material} {mem.tool_used} {mem.notes}"
            all_words.update(self._tokenize(text))
        
        self.vocabulary = sorted(list(all_words))
        self.word_to_idx = {w: i for i, w in enumerate(self.vocabulary)}
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to vector."""
        if not self.vocabulary:
            return np.zeros(1)
        
        vec = np.zeros(len(self.vocabulary))
        words = self._tokenize(text)
        
        for word in words:
            if word in self.word_to_idx:
                vec[self.word_to_idx[word]] = 1.0
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    def _rebuild_vectors(self):
        """Rebuild all memory vectors."""
        self._build_vocabulary()
        
        vectors = []
        for mem in self.memories:
            text = f"{mem.summary} {mem.part_type} {mem.defect_type} {mem.material} {mem.tool_used} {mem.notes}"
            vectors.append(self._text_to_vector(text))
        
        self.vectors = np.array(vectors) if vectors else None
    
    def add(self, memory: RepairMemory):
        """Add a memory to the store."""
        self.memories.append(memory)
        self._rebuild_vectors()
        
        if self.persistence_path:
            self._save()
    
    def search(self, query: str, top_k: int = 3) -> List[RecallResult]:
        """
        Search for similar memories.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            List of RecallResult ordered by similarity
        """
        if not self.memories or self.vectors is None:
            return []
        
        query_vec = self._text_to_vector(query)
        
        # Cosine similarities
        similarities = np.dot(self.vectors, query_vec)
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum threshold
                results.append(RecallResult(
                    memory=self.memories[idx],
                    similarity=float(similarities[idx])
                ))
        
        return results
    
    def _save(self):
        """Persist memories to disk."""
        if not self.persistence_path:
            return
        
        data = [m.to_dict() for m in self.memories]
        with open(self.persistence_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        """Load memories from disk."""
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return
        
        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)
            
            self.memories = [RepairMemory.from_dict(d) for d in data]
            self._rebuild_vectors()
        except Exception as e:
            print(f"Warning: Failed to load memory bank: {e}")
    
    def clear(self):
        """Clear all memories."""
        self.memories = []
        self.vectors = None
        
        if self.persistence_path and os.path.exists(self.persistence_path):
            os.remove(self.persistence_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory bank statistics."""
        if not self.memories:
            return {"total_memories": 0}
        
        defect_types = {}
        part_types = {}
        
        for m in self.memories:
            if m.defect_type:
                defect_types[m.defect_type] = defect_types.get(m.defect_type, 0) + 1
            if m.part_type:
                part_types[m.part_type] = part_types.get(m.part_type, 0) + 1
        
        return {
            "total_memories": len(self.memories),
            "defect_types": defect_types,
            "part_types": part_types,
            "vocabulary_size": len(self.vocabulary)
        }


# ============ GLOBAL INSTANCE ============

_memory_store: Optional[MemoryVectorStore] = None


def get_memory_store() -> MemoryVectorStore:
    """Get or create the memory store singleton."""
    global _memory_store
    if _memory_store is None:
        # Default persistence path
        persistence_path = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', 'temp', 'memory_bank.json'
        )
        _memory_store = MemoryVectorStore(persistence_path)
    return _memory_store


# ============ MAIN API ============

def store_repair(
    summary: str,
    part_type: str = "",
    defect_type: str = "",
    material: str = "",
    duration_seconds: float = 0.0,
    tool_used: str = "",
    success: bool = True,
    notes: str = ""
) -> RepairMemory:
    """
    Store a completed repair in memory.
    
    Args:
        summary: Brief description of the repair
        part_type: Type of part (turbine, pipe, etc.)
        defect_type: Type of defect (rust, crack, etc.)
        material: Material (steel, aluminum, etc.)
        duration_seconds: How long the repair took
        tool_used: Primary tool used
        success: Whether repair was successful
        notes: Additional notes
        
    Returns:
        The stored RepairMemory object
    """
    # Generate unique ID
    id_content = f"{summary}{datetime.now().isoformat()}"
    memory_id = hashlib.md5(id_content.encode()).hexdigest()[:8]
    
    memory = RepairMemory(
        id=memory_id,
        summary=summary,
        timestamp=datetime.now().isoformat(),
        part_type=part_type,
        defect_type=defect_type,
        material=material,
        duration_seconds=duration_seconds,
        tool_used=tool_used,
        success=success,
        notes=notes
    )
    
    store = get_memory_store()
    store.add(memory)
    
    return memory


def recall_similar_repairs(query: str, top_k: int = 3) -> List[RecallResult]:
    """
    Recall similar past repairs.
    
    Args:
        query: Natural language query about a repair situation
        top_k: Number of results to return
        
    Returns:
        List of RecallResult with similar past repairs
        
    Example:
        >>> results = recall_similar_repairs("How to fix rust on steel?")
        >>> for r in results:
        ...     print(r.get_summary())
    """
    store = get_memory_store()
    return store.search(query, top_k)


def get_memory_stats() -> Dict[str, Any]:
    """Get statistics about stored memories."""
    return get_memory_store().get_stats()


def clear_all_memories():
    """Clear all stored memories."""
    get_memory_store().clear()


# ============ TESTING ============

if __name__ == "__main__":
    print("=" * 60)
    print("Memory Bank - Test")
    print("=" * 60)
    
    # Clear existing for clean test
    clear_all_memories()
    
    # Store some sample repairs
    print("\nStoring sample repairs...")
    
    store_repair(
        summary="Ground rust off steel turbine blade using 3000 RPM",
        part_type="turbine_blade",
        defect_type="rust",
        material="steel",
        duration_seconds=180,
        tool_used="grinder"
    )
    
    store_repair(
        summary="Sanded aluminum panel with 80-grit disc at 1200 RPM",
        part_type="panel",
        defect_type="scratch",
        material="aluminum", 
        duration_seconds=60,
        tool_used="sander"
    )
    
    store_repair(
        summary="Filled crack in composite wing with epoxy filler",
        part_type="wing",
        defect_type="crack",
        material="composite",
        duration_seconds=300,
        tool_used="filler_applicator"
    )
    
    print(f"\nStats: {get_memory_stats()}")
    
    # Test recall
    print("\nRecalling similar repairs for 'rust on metal part'...")
    results = recall_similar_repairs("rust on metal part")
    
    for r in results:
        print(f"\n{r.get_summary()}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
