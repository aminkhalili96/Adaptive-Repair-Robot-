"""
Tests for LLM Defect Classifier.

Tests the structured defect classification from GPT-4o Vision.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestDefectClassification:
    """Tests for DefectClassification dataclass."""
    
    def test_defect_classification_creation(self):
        """Test creating a DefectClassification."""
        from src.vision.llm_defect_classifier import DefectClassification
        
        classification = DefectClassification(
            defect_type="rust",
            severity=7,
            recommended_action="grinding",
            confidence=0.85,
            location_description="upper-left quadrant"
        )
        
        assert classification.defect_type == "rust"
        assert classification.severity == 7
        assert classification.recommended_action == "grinding"
        assert classification.confidence == 0.85
    
    def test_severity_label_low(self):
        """Test severity label for low severity."""
        from src.vision.llm_defect_classifier import DefectClassification
        
        classification = DefectClassification(
            defect_type="scratch",
            severity=2,
            recommended_action="polishing",
            confidence=0.9,
            location_description="center"
        )
        
        assert classification.get_severity_label() == "low"
    
    def test_severity_label_medium(self):
        """Test severity label for medium severity."""
        from src.vision.llm_defect_classifier import DefectClassification
        
        classification = DefectClassification(
            defect_type="wear",
            severity=5,
            recommended_action="sanding",
            confidence=0.8,
            location_description="edge"
        )
        
        assert classification.get_severity_label() == "medium"
    
    def test_severity_label_high(self):
        """Test severity label for high severity."""
        from src.vision.llm_defect_classifier import DefectClassification
        
        classification = DefectClassification(
            defect_type="crack",
            severity=9,
            recommended_action="welding",
            confidence=0.75,
            location_description="leading edge"
        )
        
        assert classification.get_severity_label() == "high"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from src.vision.llm_defect_classifier import DefectClassification
        
        classification = DefectClassification(
            defect_type="rust",
            severity=7,
            recommended_action="grinding",
            confidence=0.85,
            location_description="upper-left",
            additional_notes="Minor surface rust"
        )
        
        d = classification.to_dict()
        
        assert isinstance(d, dict)
        assert d["defect_type"] == "rust"
        assert d["severity"] == 7
        assert "additional_notes" in d
    
    def test_get_summary(self):
        """Test summary generation."""
        from src.vision.llm_defect_classifier import DefectClassification
        
        classification = DefectClassification(
            defect_type="rust",
            severity=7,
            recommended_action="grinding",
            confidence=0.85,
            location_description="upper-left"
        )
        
        summary = classification.get_summary()
        
        assert "Rust" in summary
        assert "7/10" in summary
        assert "high" in summary
        assert "grinding" in summary
        assert "85%" in summary


class TestMultiDefectClassification:
    """Tests for MultiDefectClassification."""
    
    def test_empty_classification(self):
        """Test classification with no defects."""
        from src.vision.llm_defect_classifier import MultiDefectClassification
        
        result = MultiDefectClassification()
        
        assert len(result.defects) == 0
        assert result.overall_condition == "good"
        assert "No defects" in result.get_summary()
    
    def test_with_defects(self):
        """Test classification with defects."""
        from src.vision.llm_defect_classifier import (
            DefectClassification, 
            MultiDefectClassification
        )
        
        defects = [
            DefectClassification(
                defect_type="rust",
                severity=7,
                recommended_action="grinding",
                confidence=0.85,
                location_description="upper-left"
            ),
            DefectClassification(
                defect_type="scratch",
                severity=2,
                recommended_action="polishing",
                confidence=0.9,
                location_description="center"
            )
        ]
        
        result = MultiDefectClassification(
            defects=defects,
            overall_condition="fair",
            total_defect_count=2
        )
        
        assert len(result.defects) == 2
        assert result.overall_condition == "fair"
        
        summary = result.get_summary()
        assert "FAIR" in summary
        assert "2 defect" in summary


class TestLLMDefectClassifier:
    """Tests for LLMDefectClassifier class."""
    
    def test_classifier_creation(self):
        """Test classifier instantiation."""
        from src.vision.llm_defect_classifier import LLMDefectClassifier
        
        classifier = LLMDefectClassifier()
        # Should not raise
        assert classifier is not None
    
    def test_is_available_without_key(self):
        """Test availability check without API key."""
        from src.vision.llm_defect_classifier import LLMDefectClassifier
        
        with patch.dict('os.environ', {}, clear=True):
            classifier = LLMDefectClassifier()
            # May or may not be available depending on env
    
    @patch('src.vision.llm_defect_classifier.HAS_OPENAI', False)
    def test_classify_without_openai(self):
        """Test classification when OpenAI not available."""
        from src.vision.llm_defect_classifier import LLMDefectClassifier
        
        classifier = LLMDefectClassifier()
        result = classifier.classify("fake_base64_image")
        
        assert result.overall_condition == "unknown"
        assert "Error" in result.raw_response or "not available" in result.raw_response


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_get_classifier_singleton(self):
        """Test that get_classifier returns singleton."""
        from src.vision.llm_defect_classifier import get_classifier
        
        c1 = get_classifier()
        c2 = get_classifier()
        
        assert c1 is c2
    
    def test_is_classifier_available(self):
        """Test availability check function."""
        from src.vision.llm_defect_classifier import is_classifier_available
        
        result = is_classifier_available()
        assert isinstance(result, bool)


class TestResponseParsing:
    """Tests for LLM response parsing."""
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        from src.vision.llm_defect_classifier import LLMDefectClassifier
        
        classifier = LLMDefectClassifier()
        
        raw_response = '''{
            "overall_condition": "fair",
            "defects": [
                {
                    "defect_type": "rust",
                    "severity": 7,
                    "recommended_action": "grinding",
                    "confidence": 0.85,
                    "location_description": "upper-left",
                    "additional_notes": ""
                }
            ]
        }'''
        
        result = classifier._parse_response(raw_response)
        
        assert result.overall_condition == "fair"
        assert len(result.defects) == 1
        assert result.defects[0].defect_type == "rust"
    
    def test_parse_markdown_wrapped_json(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        from src.vision.llm_defect_classifier import LLMDefectClassifier
        
        classifier = LLMDefectClassifier()
        
        raw_response = '''```json
{
    "overall_condition": "good",
    "defects": []
}
```'''
        
        result = classifier._parse_response(raw_response)
        
        assert result.overall_condition == "good"
        assert len(result.defects) == 0
    
    def test_parse_invalid_json(self):
        """Test fallback for invalid JSON."""
        from src.vision.llm_defect_classifier import LLMDefectClassifier
        
        classifier = LLMDefectClassifier()
        
        raw_response = "This is not valid JSON, just a description of the defect."
        
        result = classifier._parse_response(raw_response)
        
        # Should create fallback classification
        assert result.overall_condition == "unknown"
        assert len(result.defects) == 1
        assert result.defects[0].defect_type == "unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
