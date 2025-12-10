"""
Tests for Safety Reviewer.

Tests the LLM-powered safety validation for repair plans.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestSafetyReview:
    """Tests for SafetyReview dataclass."""
    
    def test_safety_review_creation(self):
        """Test creating a SafetyReview."""
        from src.agent.safety_reviewer import SafetyReview
        
        review = SafetyReview(
            approved=True,
            risk_level="low",
            warnings=[],
            recommendations=["Consider reducing RPM"],
            checked_items=["Workspace bounds", "RPM limits"]
        )
        
        assert review.approved is True
        assert review.risk_level == "low"
        assert len(review.checked_items) == 2
    
    def test_summary_approved(self):
        """Test summary for approved plan."""
        from src.agent.safety_reviewer import SafetyReview
        
        review = SafetyReview(
            approved=True,
            risk_level="low"
        )
        
        summary = review.get_summary()
        
        assert "✅" in summary
        assert "APPROVED" in summary
        assert "🟢" in summary
        assert "LOW" in summary
    
    def test_summary_rejected(self):
        """Test summary for rejected plan."""
        from src.agent.safety_reviewer import SafetyReview
        
        review = SafetyReview(
            approved=False,
            risk_level="high",
            warnings=["RPM exceeds limit", "Incompatible tool"]
        )
        
        summary = review.get_summary()
        
        assert "⚠️" in summary
        assert "REVIEW REQUIRED" in summary
        assert "🟠" in summary
        assert "HIGH" in summary
        assert "RPM exceeds limit" in summary
    
    def test_summary_with_recommendations(self):
        """Test summary includes recommendations."""
        from src.agent.safety_reviewer import SafetyReview
        
        review = SafetyReview(
            approved=True,
            risk_level="medium",
            recommendations=["Consider lower speed", "Add coolant"]
        )
        
        summary = review.get_summary()
        
        assert "Recommendations" in summary
        assert "💡" in summary
        assert "Consider lower speed" in summary


class TestSafetyReviewer:
    """Tests for SafetyReviewer class."""
    
    def test_reviewer_creation(self):
        """Test reviewer instantiation."""
        from src.agent.safety_reviewer import SafetyReviewer
        
        reviewer = SafetyReviewer()
        assert reviewer is not None
    
    def test_heuristic_review_safe_plan(self):
        """Test heuristic review for a safe plan."""
        from src.agent.safety_reviewer import SafetyReviewer
        
        reviewer = SafetyReviewer()
        
        safe_plan = {
            "defects": [
                {"type": "rust", "severity": "medium", "position": (0.5, 0.1, 0.3)}
            ],
            "material": "steel",
            "tool": "grinder",
            "rpm": 2500
        }
        
        result = reviewer._heuristic_review(safe_plan)
        
        assert result.approved is True
        assert result.risk_level == "low"
        assert "Workspace bounds" in result.checked_items
    
    def test_heuristic_review_high_rpm(self):
        """Test heuristic review catches high RPM."""
        from src.agent.safety_reviewer import SafetyReviewer
        
        reviewer = SafetyReviewer()
        
        unsafe_plan = {
            "defects": [],
            "material": "aluminum",
            "tool": "sander",
            "rpm": 2000  # Exceeds aluminum limit of 1500
        }
        
        result = reviewer._heuristic_review(unsafe_plan)
        
        assert result.approved is False
        assert result.risk_level == "high"
        assert any("RPM" in w for w in result.warnings)
    
    def test_heuristic_review_incompatible_tool(self):
        """Test heuristic review catches incompatible tool."""
        from src.agent.safety_reviewer import SafetyReviewer
        
        reviewer = SafetyReviewer()
        
        bad_tool_plan = {
            "defects": [],
            "material": "composite",
            "tool": "grinder",  # Not suitable for composite
            "rpm": 800
        }
        
        result = reviewer._heuristic_review(bad_tool_plan)
        
        assert result.approved is False
        assert any("composite" in w.lower() or "grinder" in w.lower() for w in result.warnings)
    
    def test_heuristic_review_out_of_bounds(self):
        """Test heuristic review catches position out of bounds."""
        from src.agent.safety_reviewer import SafetyReviewer
        
        reviewer = SafetyReviewer()
        
        out_of_bounds_plan = {
            "defects": [
                {"position": (0.1, 0, 0.3)}  # X < 0.2
            ],
            "material": "steel",
            "tool": "grinder",
            "rpm": 2500
        }
        
        result = reviewer._heuristic_review(out_of_bounds_plan)
        
        assert result.risk_level != "low"
        assert any("position" in w.lower() or "bounds" in w.lower() for w in result.warnings)
    
    @patch('src.agent.safety_reviewer.HAS_OPENAI', False)
    def test_review_falls_back_to_heuristic(self):
        """Test that review falls back to heuristic when OpenAI unavailable."""
        from src.agent.safety_reviewer import SafetyReviewer
        
        reviewer = SafetyReviewer()
        reviewer.client = None  # Simulate no API
        
        plan = {
            "defects": [],
            "material": "steel",
            "tool": "grinder",
            "rpm": 2500
        }
        
        result = reviewer.review(plan)
        
        # Should still get a result from heuristic
        assert result is not None
        assert "Workspace bounds" in result.checked_items


class TestResponseParsing:
    """Tests for LLM response parsing."""
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        from src.agent.safety_reviewer import SafetyReviewer
        
        reviewer = SafetyReviewer()
        
        raw_response = '''{
            "approved": true,
            "risk_level": "low",
            "warnings": [],
            "recommendations": ["Consider reducing speed"],
            "checked_items": ["Bounds", "RPM"]
        }'''
        
        result = reviewer._parse_response(raw_response)
        
        assert result.approved is True
        assert result.risk_level == "low"
        assert len(result.recommendations) == 1
    
    def test_parse_markdown_wrapped_json(self):
        """Test parsing JSON wrapped in markdown."""
        from src.agent.safety_reviewer import SafetyReviewer
        
        reviewer = SafetyReviewer()
        
        raw_response = '''```json
{
    "approved": false,
    "risk_level": "high",
    "warnings": ["RPM too high"],
    "recommendations": [],
    "checked_items": []
}
```'''
        
        result = reviewer._parse_response(raw_response)
        
        assert result.approved is False
        assert result.risk_level == "high"
    
    def test_parse_invalid_json(self):
        """Test fallback for invalid JSON."""
        from src.agent.safety_reviewer import SafetyReviewer
        
        reviewer = SafetyReviewer()
        
        raw_response = "Not valid JSON"
        
        result = reviewer._parse_response(raw_response)
        
        # Should be conservative (not approved)
        assert result.approved is False
        assert result.risk_level == "medium"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_get_reviewer_singleton(self):
        """Test that get_reviewer returns singleton."""
        from src.agent.safety_reviewer import get_reviewer
        
        r1 = get_reviewer()
        r2 = get_reviewer()
        
        assert r1 is r2
    
    def test_is_reviewer_available(self):
        """Test availability check function."""
        from src.agent.safety_reviewer import is_reviewer_available
        
        result = is_reviewer_available()
        assert isinstance(result, bool)
    
    def test_review_plan_safety_function(self):
        """Test the main review_plan_safety function."""
        from src.agent.safety_reviewer import review_plan_safety
        
        plan = {
            "defects": [
                {"type": "rust", "severity": "medium", "position": (0.5, 0.1, 0.3)}
            ],
            "material": "steel",
            "tool": "grinder",
            "rpm": 2500
        }
        
        result = review_plan_safety(plan)
        
        assert result is not None
        assert hasattr(result, 'approved')
        assert hasattr(result, 'risk_level')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
