"""
LLM Defect Classifier - Multimodal Defect Diagnosis.

Uses GPT-4o Vision to provide structured defect classifications 
from images of industrial parts.

Usage:
    from src.vision.llm_defect_classifier import classify_defect_from_image
    
    result = classify_defect_from_image(image_base64)
    # Returns: DefectClassification with type, severity, recommended_action, confidence
"""

import os
import json
from typing import Optional, List
from dataclasses import dataclass, field

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============ DATA STRUCTURES ============

@dataclass
class DefectClassification:
    """Structured classification result from GPT-4o Vision."""
    defect_type: str  # rust, crack, dent, erosion, pitting, corrosion, wear, none
    severity: int  # 1-10 scale
    recommended_action: str  # grinding, sanding, filling, coating, etc.
    confidence: float  # 0.0-1.0
    location_description: str  # "upper-left quadrant", "leading edge", etc.
    additional_notes: str = ""
    raw_response: str = ""
    
    def to_dict(self) -> dict:
        return {
            "defect_type": self.defect_type,
            "severity": self.severity,
            "recommended_action": self.recommended_action,
            "confidence": self.confidence,
            "location_description": self.location_description,
            "additional_notes": self.additional_notes
        }
    
    def get_severity_label(self) -> str:
        if self.severity <= 3:
            return "low"
        elif self.severity <= 6:
            return "medium"
        else:
            return "high"
    
    def get_summary(self) -> str:
        return (
            f"**{self.defect_type.title()}** (Severity: {self.severity}/10 - {self.get_severity_label()})\n"
            f"Location: {self.location_description}\n"
            f"Recommended: {self.recommended_action}\n"
            f"Confidence: {self.confidence:.0%}"
        )


@dataclass 
class MultiDefectClassification:
    """Multiple defects classified from a single image."""
    defects: List[DefectClassification] = field(default_factory=list)
    overall_condition: str = "good"  # good, fair, poor, critical
    total_defect_count: int = 0
    raw_response: str = ""
    
    def get_summary(self) -> str:
        if not self.defects:
            return "No defects detected. Part appears to be in good condition."
        
        lines = [f"**Overall Condition: {self.overall_condition.upper()}**\n"]
        lines.append(f"Detected {len(self.defects)} defect(s):\n")
        
        for i, defect in enumerate(self.defects, 1):
            lines.append(f"{i}. {defect.get_summary()}\n")
        
        return "\n".join(lines)


# ============ SYSTEM PROMPT ============

CLASSIFIER_SYSTEM_PROMPT = """You are an expert industrial defect classifier for manufacturing quality control.

Analyze the provided image of an industrial part and identify all visible defects.

For EACH defect found, provide a JSON object with:
- defect_type: one of ["rust", "crack", "dent", "erosion", "pitting", "corrosion", "wear", "scratch", "none"]
- severity: integer 1-10 (1=minor cosmetic, 5=moderate functional concern, 10=critical structural failure)
- recommended_action: one of ["grinding", "sanding", "filling", "coating", "polishing", "welding", "replacement", "no_action"]
- confidence: float 0.0-1.0 (your confidence in this classification)
- location_description: describe WHERE on the part (e.g., "upper-left quadrant", "leading edge", "center")
- additional_notes: any relevant observations

If NO defects are found, return a single defect with defect_type="none" and severity=0.

RESPOND WITH ONLY A JSON OBJECT in this exact format:
{
    "overall_condition": "good|fair|poor|critical",
    "defects": [
        {
            "defect_type": "...",
            "severity": N,
            "recommended_action": "...",
            "confidence": 0.X,
            "location_description": "...",
            "additional_notes": "..."
        }
    ]
}

Be precise and technical. Base severity on visible damage extent and potential structural impact."""


# ============ CLASSIFIER CLASS ============

class LLMDefectClassifier:
    """
    GPT-4o Vision-based defect classifier.
    
    Provides structured JSON output for defect detection,
    suitable for automated pipeline integration.
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
        """Check if the classifier is ready to use."""
        return self.client is not None
    
    def classify(self, image_base64: str) -> MultiDefectClassification:
        """
        Classify defects in an image.
        
        Args:
            image_base64: Base64-encoded PNG/JPEG image
            
        Returns:
            MultiDefectClassification with all detected defects
        """
        if not self.is_available:
            return MultiDefectClassification(
                overall_condition="unknown",
                raw_response="Error: OpenAI API not available. Please check API key."
            )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this industrial part for defects."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                            }
                        ]
                    }
                ],
                max_tokens=800,
                temperature=0.3  # Lower temperature for more consistent classification
            )
            
            raw_content = response.choices[0].message.content
            return self._parse_response(raw_content)
            
        except Exception as e:
            return MultiDefectClassification(
                overall_condition="error",
                raw_response=f"Classification failed: {str(e)}"
            )
    
    def _parse_response(self, raw_content: str) -> MultiDefectClassification:
        """Parse LLM response into structured classification."""
        try:
            # Try to extract JSON from the response
            # Handle cases where LLM might wrap in markdown code blocks
            content = raw_content.strip()
            if content.startswith("```"):
                # Remove markdown code blocks
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            data = json.loads(content)
            
            defects = []
            for d in data.get("defects", []):
                defects.append(DefectClassification(
                    defect_type=d.get("defect_type", "unknown"),
                    severity=int(d.get("severity", 5)),
                    recommended_action=d.get("recommended_action", "inspection"),
                    confidence=float(d.get("confidence", 0.5)),
                    location_description=d.get("location_description", "unknown"),
                    additional_notes=d.get("additional_notes", ""),
                    raw_response=""
                ))
            
            return MultiDefectClassification(
                defects=defects,
                overall_condition=data.get("overall_condition", "unknown"),
                total_defect_count=len(defects),
                raw_response=raw_content
            )
            
        except json.JSONDecodeError:
            # Fallback: create a single defect from free-form text
            return MultiDefectClassification(
                defects=[DefectClassification(
                    defect_type="unknown",
                    severity=5,
                    recommended_action="manual_inspection",
                    confidence=0.3,
                    location_description="see description",
                    additional_notes=raw_content,
                    raw_response=raw_content
                )],
                overall_condition="unknown",
                total_defect_count=1,
                raw_response=raw_content
            )


# ============ CONVENIENCE FUNCTIONS ============

_classifier_instance: Optional[LLMDefectClassifier] = None


def get_classifier() -> LLMDefectClassifier:
    """Get or create the classifier singleton."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = LLMDefectClassifier()
    return _classifier_instance


def classify_defect_from_image(image_base64: str) -> MultiDefectClassification:
    """
    Main API: Classify defects in an image.
    
    Args:
        image_base64: Base64-encoded image data
        
    Returns:
        MultiDefectClassification with structured defect data
        
    Example:
        >>> result = classify_defect_from_image(base64_data)
        >>> print(result.get_summary())
        Overall Condition: FAIR
        Detected 2 defect(s):
        1. Rust (Severity: 7/10 - high)
           Location: upper-left quadrant
           Recommended: grinding
        2. Scratch (Severity: 2/10 - low)
           Location: center
           Recommended: polishing
    """
    classifier = get_classifier()
    return classifier.classify(image_base64)


def is_classifier_available() -> bool:
    """Check if the LLM classifier is available."""
    return get_classifier().is_available


# ============ TESTING ============

if __name__ == "__main__":
    print("=" * 60)
    print("LLM Defect Classifier - Status Check")
    print("=" * 60)
    
    classifier = get_classifier()
    
    if classifier.is_available:
        print("✓ OpenAI API available")
        print(f"  Model: {classifier.model}")
    else:
        print("✗ OpenAI API NOT available")
        print("  Set OPENAI_API_KEY environment variable")
    
    print("\nTo test with an actual image:")
    print("  import base64")
    print("  with open('test.png', 'rb') as f:")
    print("      b64 = base64.b64encode(f.read()).decode()")
    print("  result = classify_defect_from_image(b64)")
    print("  print(result.get_summary())")
