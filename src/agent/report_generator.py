"""
Quality Report Generator - LLM-Powered Audit Reports.

Generates human-readable, auditor-friendly quality reports after repairs
using GPT-4o to transform repair logs into formatted documentation.

Usage:
    from src.agent.report_generator import generate_quality_report
    
    report = generate_quality_report(repair_log, defects, part_info)
    # Returns: Formatted markdown report ready for PDF conversion
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============ DATA STRUCTURES ============

@dataclass
class QualityReport:
    """Generated quality report."""
    title: str
    content: str  # Markdown formatted report
    generated_at: str
    part_id: str
    operator_notes: str = ""
    is_compliant: bool = True
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def to_markdown(self) -> str:
        """Return the full markdown report."""
        return self.content
    
    def get_summary(self) -> str:
        """Get a brief summary for chat display."""
        status = "✅ COMPLIANT" if self.is_compliant else "⚠️ NON-COMPLIANT"
        return f"**Quality Report Generated**\n\nPart: {self.part_id}\nStatus: {status}\nGenerated: {self.generated_at}"


# ============ SYSTEM PROMPT ============

REPORT_GENERATOR_PROMPT = """You are a professional quality assurance documentation specialist for industrial manufacturing.

Generate a formal quality report from the provided repair data. The report must be:
- Professional and suitable for regulatory audits
- Clear and well-structured
- Include all relevant technical details
- Formatted in Markdown

Use this structure:
# Quality Assurance Report

## Summary
Brief overview of the repair operation.

## Part Information
| Field | Value |
|-------|-------|
| Part ID | ... |
| Part Type | ... |
| Material | ... |

## Defects Identified
For each defect:
- Location
- Type
- Severity
- Detection method

## Repair Actions Taken
For each repair:
- Action type
- Tool used
- Parameters (RPM, pressure, etc.)
- Duration

## Quality Verification
- Pre-repair condition
- Post-repair condition
- Compliance status

## Operator Notes
Any additional observations.

## Certification
- Timestamp
- System version
- Compliance status

Be thorough but concise. Use tables where appropriate."""


# ============ REPORT GENERATOR ============

class ReportGenerator:
    """
    LLM-powered quality report generator.
    
    Transforms structured repair logs into formatted,
    audit-ready documentation.
    """
    
    def __init__(self):
        self.client = None
        self.model = "gpt-4o"
        
        if HAS_OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
    
    @property
    def is_available(self) -> bool:
        """Check if the generator is ready."""
        return self.client is not None
    
    def generate(
        self,
        repair_log: Dict[str, Any],
        defects: List[Dict],
        part_info: Dict[str, Any],
        operator_notes: str = ""
    ) -> QualityReport:
        """
        Generate a quality report from repair data.
        
        Args:
            repair_log: Dict with repair actions, durations, tools used
            defects: List of defect dicts with type, severity, position
            part_info: Dict with part_id, mesh_name, material
            operator_notes: Optional free-form notes
            
        Returns:
            QualityReport with formatted markdown content
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        part_id = part_info.get("part_id", f"PART-{datetime.now().strftime('%Y%m%d%H%M')}")
        
        # Build input data for LLM
        input_data = {
            "part_info": part_info,
            "defects": defects,
            "repair_log": repair_log,
            "operator_notes": operator_notes,
            "timestamp": timestamp,
            "system_version": "AARR v1.0"
        }
        
        if not self.is_available:
            # Generate a basic template without LLM
            return self._generate_template_report(input_data, part_id, timestamp)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REPORT_GENERATOR_PROMPT},
                    {
                        "role": "user",
                        "content": f"Generate a quality report from this data:\n\n```json\n{json.dumps(input_data, indent=2)}\n```"
                    }
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            
            # Determine compliance (simple heuristic)
            high_severity_count = sum(
                1 for d in defects 
                if d.get("severity", "medium") == "high" or d.get("severity", 5) > 7
            )
            is_compliant = high_severity_count == 0 or repair_log.get("completed", False)
            
            return QualityReport(
                title=f"Quality Report - {part_id}",
                content=content,
                generated_at=timestamp,
                part_id=part_id,
                operator_notes=operator_notes,
                is_compliant=is_compliant,
                warnings=[] if is_compliant else ["High severity defects detected"]
            )
            
        except Exception as e:
            return QualityReport(
                title=f"Quality Report - {part_id}",
                content=f"# Error Generating Report\n\nAn error occurred: {str(e)}\n\nPlease generate report manually.",
                generated_at=timestamp,
                part_id=part_id,
                is_compliant=False,
                warnings=[f"Generation error: {str(e)}"]
            )
    
    def _generate_template_report(
        self, 
        data: Dict, 
        part_id: str, 
        timestamp: str
    ) -> QualityReport:
        """Generate a basic template report without LLM."""
        part_info = data.get("part_info", {})
        defects = data.get("defects", [])
        repair_log = data.get("repair_log", {})
        
        # Build defect table
        defect_rows = []
        for i, d in enumerate(defects, 1):
            dtype = d.get("type", "unknown")
            severity = d.get("severity", "medium")
            pos = d.get("position", (0, 0, 0))
            defect_rows.append(f"| {i} | {dtype} | {severity} | {pos} |")
        
        defect_table = "\n".join(defect_rows) if defect_rows else "| - | No defects | - | - |"
        
        content = f"""# Quality Assurance Report

## Summary
Automated repair operation completed on {timestamp}.

## Part Information
| Field | Value |
|-------|-------|
| Part ID | {part_id} |
| Part Type | {part_info.get('mesh_name', 'Unknown')} |
| Material | {part_info.get('material', 'Steel')} |

## Defects Identified
| # | Type | Severity | Position |
|---|------|----------|----------|
{defect_table}

## Repair Actions Taken
- Actions: {repair_log.get('actions_count', len(defects))}
- Total Duration: {repair_log.get('total_duration', 'N/A')}
- Status: {repair_log.get('status', 'Completed')}

## Quality Verification
- Pre-repair: Defects detected via automated vision system
- Post-repair: Pending verification
- Compliance: {'✅ Compliant' if not any(d.get('severity') == 'high' for d in defects) else '⚠️ Review Required'}

## Operator Notes
{data.get('operator_notes') or 'No additional notes.'}

## Certification
- Generated: {timestamp}
- System: AARR v1.0
- Report ID: RPT-{part_id}

---
*This is an auto-generated report. Please review before distribution.*
"""
        
        return QualityReport(
            title=f"Quality Report - {part_id}",
            content=content,
            generated_at=timestamp,
            part_id=part_id,
            operator_notes=data.get("operator_notes", ""),
            is_compliant=True
        )


# ============ CONVENIENCE FUNCTIONS ============

_generator_instance: Optional[ReportGenerator] = None


def get_generator() -> ReportGenerator:
    """Get or create the generator singleton."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = ReportGenerator()
    return _generator_instance


def generate_quality_report(
    repair_log: Dict[str, Any] = None,
    defects: List[Dict] = None,
    part_info: Dict[str, Any] = None,
    operator_notes: str = ""
) -> QualityReport:
    """
    Main API: Generate a quality report.
    
    Args:
        repair_log: Repair actions taken
        defects: List of defects found
        part_info: Part metadata
        operator_notes: Additional notes
        
    Returns:
        QualityReport with formatted markdown
    """
    generator = get_generator()
    return generator.generate(
        repair_log=repair_log or {},
        defects=defects or [],
        part_info=part_info or {},
        operator_notes=operator_notes
    )


def is_generator_available() -> bool:
    """Check if LLM report generation is available."""
    return get_generator().is_available


# ============ TESTING ============

if __name__ == "__main__":
    print("=" * 60)
    print("Quality Report Generator - Test")
    print("=" * 60)
    
    # Sample data
    test_defects = [
        {"type": "rust", "severity": "high", "position": (0.2, 0.1, 0.3)},
        {"type": "scratch", "severity": "low", "position": (0.5, 0.2, 0.1)},
    ]
    
    test_repair_log = {
        "actions_count": 2,
        "total_duration": "4.5 minutes",
        "status": "Completed",
        "completed": True
    }
    
    test_part_info = {
        "part_id": "TB-2024-001",
        "mesh_name": "Turbine Blade",
        "material": "Titanium Alloy"
    }
    
    report = generate_quality_report(
        repair_log=test_repair_log,
        defects=test_defects,
        part_info=test_part_info,
        operator_notes="Routine maintenance check."
    )
    
    print(f"\n{report.get_summary()}\n")
    print("-" * 60)
    print(report.to_markdown())
