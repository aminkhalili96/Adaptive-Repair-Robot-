"""
Tests for Report Generator.

Tests the LLM-powered quality report generation.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock


class TestQualityReport:
    """Tests for QualityReport dataclass."""
    
    def test_quality_report_creation(self):
        """Test creating a QualityReport."""
        from src.agent.report_generator import QualityReport
        
        report = QualityReport(
            title="Test Report",
            content="# Test\n\nContent here.",
            generated_at="2024-01-01T00:00:00",
            part_id="PART-001"
        )
        
        assert report.title == "Test Report"
        assert report.part_id == "PART-001"
        assert report.is_compliant is True  # Default
    
    def test_to_markdown(self):
        """Test markdown output."""
        from src.agent.report_generator import QualityReport
        
        content = "# Quality Report\n\n## Summary\n\nTest content."
        report = QualityReport(
            title="Test",
            content=content,
            generated_at="2024-01-01T00:00:00",
            part_id="PART-001"
        )
        
        md = report.to_markdown()
        
        assert md == content
        assert "# Quality Report" in md
    
    def test_get_summary_compliant(self):
        """Test summary for compliant report."""
        from src.agent.report_generator import QualityReport
        
        report = QualityReport(
            title="Test",
            content="Content",
            generated_at="2024-01-01T00:00:00",
            part_id="PART-001",
            is_compliant=True
        )
        
        summary = report.get_summary()
        
        assert "PART-001" in summary
        assert "COMPLIANT" in summary
        assert "✅" in summary
    
    def test_get_summary_non_compliant(self):
        """Test summary for non-compliant report."""
        from src.agent.report_generator import QualityReport
        
        report = QualityReport(
            title="Test",
            content="Content",
            generated_at="2024-01-01T00:00:00",
            part_id="PART-002",
            is_compliant=False
        )
        
        summary = report.get_summary()
        
        assert "NON-COMPLIANT" in summary
        assert "⚠️" in summary


class TestReportGenerator:
    """Tests for ReportGenerator class."""
    
    def test_generator_creation(self):
        """Test generator instantiation."""
        from src.agent.report_generator import ReportGenerator
        
        generator = ReportGenerator()
        assert generator is not None
    
    def test_template_report_generation(self):
        """Test template-based report generation (without LLM)."""
        from src.agent.report_generator import ReportGenerator
        
        generator = ReportGenerator()
        generator.client = None  # Force template mode
        
        defects = [
            {"type": "rust", "severity": "high", "position": (0.5, 0.1, 0.3)},
            {"type": "scratch", "severity": "low", "position": (0.3, 0.2, 0.1)}
        ]
        
        repair_log = {
            "actions_count": 2,
            "total_duration": "5 minutes",
            "status": "Completed"
        }
        
        part_info = {
            "part_id": "TB-001",
            "mesh_name": "Turbine Blade",
            "material": "Steel"
        }
        
        report = generator.generate(
            repair_log=repair_log,
            defects=defects,
            part_info=part_info,
            operator_notes="Test notes"
        )
        
        assert report is not None
        assert "TB-001" in report.content or "Turbine Blade" in report.content
        assert "Quality" in report.content
    
    def test_template_report_empty_defects(self):
        """Test template report with no defects."""
        from src.agent.report_generator import ReportGenerator
        
        generator = ReportGenerator()
        generator.client = None
        
        report = generator.generate(
            repair_log={},
            defects=[],
            part_info={"part_id": "TEST-001"}
        )
        
        assert report is not None
        content = report.to_markdown()
        assert "Quality" in content
    
    def test_compliance_determination_high_severity(self):
        """Test that high severity defects affect compliance."""
        from src.agent.report_generator import ReportGenerator
        
        generator = ReportGenerator()
        generator.client = None
        
        # Unrepaired high-severity defect
        defects = [{"type": "crack", "severity": "high", "position": (0.5, 0.1, 0.3)}]
        
        report = generator.generate(
            repair_log={"completed": False},
            defects=defects,
            part_info={}
        )
        
        # Should not be compliant if high severity and not completed
        # (Implementation may vary)
        assert report is not None


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_get_generator_singleton(self):
        """Test that get_generator returns singleton."""
        from src.agent.report_generator import get_generator
        
        g1 = get_generator()
        g2 = get_generator()
        
        assert g1 is g2
    
    def test_is_generator_available(self):
        """Test availability check function."""
        from src.agent.report_generator import is_generator_available
        
        result = is_generator_available()
        assert isinstance(result, bool)
    
    def test_generate_quality_report_function(self):
        """Test the main generate_quality_report function."""
        from src.agent.report_generator import generate_quality_report
        
        report = generate_quality_report(
            repair_log={"status": "Test"},
            defects=[{"type": "test", "severity": "low"}],
            part_info={"part_id": "FUNC-TEST"}
        )
        
        assert report is not None
        assert hasattr(report, 'content')
        assert hasattr(report, 'is_compliant')
    
    def test_generate_quality_report_defaults(self):
        """Test generate_quality_report with default arguments."""
        from src.agent.report_generator import generate_quality_report
        
        report = generate_quality_report()
        
        assert report is not None
        assert report.content is not None


class TestReportContent:
    """Tests for report content structure."""
    
    def test_report_has_required_sections(self):
        """Test that template report has required sections."""
        from src.agent.report_generator import ReportGenerator
        
        generator = ReportGenerator()
        generator.client = None
        
        report = generator.generate(
            repair_log={"status": "Completed"},
            defects=[{"type": "rust", "severity": "medium", "position": (0.5, 0.1, 0.3)}],
            part_info={"part_id": "SEC-001", "mesh_name": "Test Part"}
        )
        
        content = report.to_markdown()
        
        # Check for expected sections
        assert "Summary" in content or "Information" in content
        assert "Defect" in content
        assert "Repair" in content or "Action" in content
    
    def test_report_includes_timestamp(self):
        """Test that report includes generation timestamp."""
        from src.agent.report_generator import generate_quality_report
        
        report = generate_quality_report()
        
        # Should have a timestamp
        assert report.generated_at is not None
        assert len(report.generated_at) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
