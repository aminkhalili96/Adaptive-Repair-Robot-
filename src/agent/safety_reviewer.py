"""
Safety Reviewer - LLM-Powered Safety Validation.

Provides a second LLM pass to validate repair plans before execution,
checking for safety violations, parameter limits, and material compatibility.

Usage:
    from src.agent.safety_reviewer import review_plan_safety
    
    result = review_plan_safety(plan)
    if not result.approved:
        print(f"Warnings: {result.warnings}")
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============ DATA STRUCTURES ============

@dataclass
class SafetyReview:
    """Result of a safety review."""
    approved: bool
    risk_level: str  # low, medium, high, critical
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    checked_items: List[str] = field(default_factory=list)
    raw_response: str = ""
    
    def get_summary(self) -> str:
        """Get formatted summary for display."""
        status_icon = "✅" if self.approved else "⚠️"
        risk_icons = {
            "low": "🟢",
            "medium": "🟡", 
            "high": "🟠",
            "critical": "🔴"
        }
        risk_icon = risk_icons.get(self.risk_level, "⚪")
        
        lines = [
            f"{status_icon} **Safety Review**: {'APPROVED' if self.approved else 'REVIEW REQUIRED'}",
            f"{risk_icon} Risk Level: **{self.risk_level.upper()}**",
        ]
        
        if self.warnings:
            lines.append("\n**Warnings:**")
            for w in self.warnings:
                lines.append(f"- ⚠️ {w}")
        
        if self.recommendations:
            lines.append("\n**Recommendations:**")
            for r in self.recommendations:
                lines.append(f"- 💡 {r}")
        
        return "\n".join(lines)


# ============ SYSTEM PROMPT ============

SAFETY_REVIEW_PROMPT = """You are a safety engineer reviewing industrial repair plans for compliance and risk.

Analyze the provided repair plan and check for:

1. **Workspace Bounds**: All positions must be within safe limits
   - X: 0.2 to 0.8 meters
   - Y: -0.4 to 0.4 meters
   - Z: 0.05 to 0.6 meters

2. **RPM Limits by Material**:
   - Aluminum: max 1500 RPM
   - Steel: max 4000 RPM
   - Composite: max 1000 RPM

3. **Pressure Limits**:
   - Low: < 50 N
   - Medium: 50-100 N
   - High: > 100 N (requires approval for composites)

4. **Tool Compatibility**:
   - Grinder: suitable for steel, not composites
   - Sander: suitable for all materials
   - Polisher: suitable for composites and aluminum

5. **Path Safety**:
   - No sharp movements (velocity < 0.5 m/s)
   - Minimum 1cm from edges
   - No collisions with fixtures

RESPOND WITH ONLY A JSON OBJECT:
{
    "approved": true/false,
    "risk_level": "low|medium|high|critical",
    "warnings": ["list of warnings"],
    "recommendations": ["list of recommendations"],
    "checked_items": ["list of items checked"]
}

Be thorough but practical. Approve if safe with warnings. Reject only for critical issues."""


# ============ SAFETY REVIEWER CLASS ============

class SafetyReviewer:
    """
    LLM-powered safety reviewer for repair plans.
    
    Provides independent validation before execution.
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
        """Check if the reviewer is ready."""
        return self.client is not None
    
    def review(self, plan: Dict[str, Any]) -> SafetyReview:
        """
        Review a repair plan for safety.
        
        Args:
            plan: Dict with repair plan details (defects, paths, params)
            
        Returns:
            SafetyReview with approval status and findings
        """
        if not self.is_available:
            # Fallback to simple heuristic checks
            return self._heuristic_review(plan)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SAFETY_REVIEW_PROMPT},
                    {
                        "role": "user",
                        "content": f"Review this repair plan for safety:\n\n```json\n{json.dumps(plan, indent=2)}\n```"
                    }
                ],
                max_tokens=500,
                temperature=0.2  # Low temperature for consistent safety checks
            )
            
            raw_content = response.choices[0].message.content
            return self._parse_response(raw_content)
            
        except Exception as e:
            return SafetyReview(
                approved=False,
                risk_level="high",
                warnings=[f"Safety review failed: {str(e)}"],
                recommendations=["Manually verify plan before execution"],
                raw_response=str(e)
            )
    
    def _parse_response(self, raw_content: str) -> SafetyReview:
        """Parse LLM response into SafetyReview."""
        try:
            # Handle markdown code blocks
            content = raw_content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            data = json.loads(content)
            
            return SafetyReview(
                approved=data.get("approved", False),
                risk_level=data.get("risk_level", "medium"),
                warnings=data.get("warnings", []),
                recommendations=data.get("recommendations", []),
                checked_items=data.get("checked_items", []),
                raw_response=raw_content
            )
            
        except json.JSONDecodeError:
            # If parsing fails, be conservative
            return SafetyReview(
                approved=False,
                risk_level="medium",
                warnings=["Could not parse safety review response"],
                recommendations=["Manual review recommended"],
                raw_response=raw_content
            )
    
    def _heuristic_review(self, plan: Dict[str, Any]) -> SafetyReview:
        """
        Simple heuristic-based safety check when LLM unavailable.
        """
        warnings = []
        recommendations = []
        checked_items = []
        risk_level = "low"
        approved = True
        
        # Check workspace bounds
        checked_items.append("Workspace bounds")
        defects = plan.get("defects", [])
        for i, d in enumerate(defects):
            pos = d.get("position", (0.5, 0, 0.3))
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                x, y, z = pos[0], pos[1], pos[2]
                if not (0.2 <= x <= 0.8):
                    warnings.append(f"Defect {i+1} X position ({x}) outside bounds")
                    risk_level = "medium"
                if not (-0.4 <= y <= 0.4):
                    warnings.append(f"Defect {i+1} Y position ({y}) outside bounds")
                    risk_level = "medium"
                if not (0.05 <= z <= 0.6):
                    warnings.append(f"Defect {i+1} Z position ({z}) outside bounds")
                    risk_level = "medium"
        
        # Check RPM limits
        checked_items.append("RPM limits")
        rpm = plan.get("rpm", 2000)
        material = plan.get("material", "steel").lower()
        
        rpm_limits = {"aluminum": 1500, "steel": 4000, "composite": 1000}
        max_rpm = rpm_limits.get(material, 3000)
        
        if rpm > max_rpm:
            warnings.append(f"RPM {rpm} exceeds {material} limit of {max_rpm}")
            risk_level = "high"
            approved = False
        elif rpm > max_rpm * 0.9:
            recommendations.append(f"Consider reducing RPM (currently at {rpm/max_rpm:.0%} of limit)")
        
        # Check tool compatibility
        checked_items.append("Tool compatibility")
        tool = plan.get("tool", "sander").lower()
        
        if material == "composite" and tool == "grinder":
            warnings.append("Grinder not recommended for composite materials")
            risk_level = "high"
            approved = False
        
        # Check for high severity defects
        checked_items.append("Defect severity")
        high_severity = sum(1 for d in defects if d.get("severity") in ["high", 8, 9, 10])
        if high_severity > 0:
            recommendations.append(f"{high_severity} high-severity defects require careful execution")
        
        return SafetyReview(
            approved=approved,
            risk_level=risk_level,
            warnings=warnings,
            recommendations=recommendations,
            checked_items=checked_items
        )


# ============ CONVENIENCE FUNCTIONS ============

_reviewer_instance: Optional[SafetyReviewer] = None


def get_reviewer() -> SafetyReviewer:
    """Get or create the reviewer singleton."""
    global _reviewer_instance
    if _reviewer_instance is None:
        _reviewer_instance = SafetyReviewer()
    return _reviewer_instance


def review_plan_safety(plan: Dict[str, Any]) -> SafetyReview:
    """
    Main API: Review a plan for safety.
    
    Args:
        plan: Repair plan dict
        
    Returns:
        SafetyReview with approval and findings
    """
    return get_reviewer().review(plan)


def is_reviewer_available() -> bool:
    """Check if LLM safety review is available."""
    return get_reviewer().is_available


# ============ TESTING ============

if __name__ == "__main__":
    print("=" * 60)
    print("Safety Reviewer - Test")
    print("=" * 60)
    
    # Test plan 1: Safe
    safe_plan = {
        "defects": [
            {"type": "rust", "severity": "medium", "position": (0.5, 0.1, 0.3)}
        ],
        "material": "steel",
        "tool": "grinder",
        "rpm": 3000
    }
    
    print("\n--- Safe Plan ---")
    result = review_plan_safety(safe_plan)
    print(result.get_summary())
    
    # Test plan 2: RPM too high
    unsafe_plan = {
        "defects": [
            {"type": "scratch", "severity": "low", "position": (0.5, 0.1, 0.3)}
        ],
        "material": "aluminum",
        "tool": "sander",
        "rpm": 2000  # Too high for aluminum
    }
    
    print("\n--- Unsafe Plan (High RPM) ---")
    result = review_plan_safety(unsafe_plan)
    print(result.get_summary())
    
    # Test plan 3: Incompatible tool
    bad_tool_plan = {
        "defects": [
            {"type": "crack", "severity": "high", "position": (0.5, 0.1, 0.3)}
        ],
        "material": "composite",
        "tool": "grinder",  # Not suitable for composite
        "rpm": 800
    }
    
    print("\n--- Bad Tool Plan ---")
    result = review_plan_safety(bad_tool_plan)
    print(result.get_summary())
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
