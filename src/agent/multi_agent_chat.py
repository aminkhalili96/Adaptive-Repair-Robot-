"""
Multi-Agent Chat System with Supervisor-Worker Architecture.

Implements a "Factory Team" of specialized agents:
- Supervisor: Routes queries, handles casual chat
- Inspector: Vision expert, controls 3D viewer
- Process Engineer: Robotics expert, plans repairs

Uses LangGraph-style state management for agent coordination.
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
import re

from src.agent.tools import get_fallback_plan, DefectInfo
from src.config import config


class AgentType(str, Enum):
    """Available agent types."""
    SUPERVISOR = "supervisor"
    INSPECTOR = "inspector"
    ENGINEER = "engineer"


class UICommandType(str, Enum):
    """Commands that agents can send to the 3D viewer."""
    HIGHLIGHT_DEFECT = "HIGHLIGHT_DEFECT"
    ZOOM_TO = "ZOOM_TO"
    SHOW_NORMALS = "SHOW_NORMALS"
    SHOW_TOOLPATH = "SHOW_TOOLPATH"
    RESET_VIEW = "RESET_VIEW"


@dataclass
class UICommand:
    """Command to update the 3D viewer."""
    type: UICommandType
    position: Optional[tuple] = None
    defect_index: Optional[int] = None
    data: Optional[Dict[str, Any]] = None


@dataclass
class AgentMessage:
    """Message from an agent."""
    agent: AgentType
    content: str
    ui_commands: List[UICommand] = field(default_factory=list)


class TeamState(TypedDict):
    """Shared state for the agent team."""
    messages: List[Dict[str, str]]
    current_speaker: AgentType
    defects: List[Dict[str, Any]]
    plans: List[Dict[str, Any]]
    ui_commands: List[Dict[str, Any]]


# Agent Avatars
AGENT_AVATARS = {
    AgentType.SUPERVISOR: "🤖",
    AgentType.INSPECTOR: "👁️",
    AgentType.ENGINEER: "🔧",
}

# Agent Names
AGENT_NAMES = {
    AgentType.SUPERVISOR: "Supervisor",
    AgentType.INSPECTOR: "Inspector",
    AgentType.ENGINEER: "Process Engineer",
}


# ============ ROUTING LOGIC ============

# Keywords that trigger specific agents
INSPECTOR_KEYWORDS = {
    "scan", "look", "inspect", "defect", "damage", "rust", "crack", "corrosion",
    "show", "find", "where", "see", "check", "severity", "high", "critical",
    "top", "bottom", "left", "right", "front", "back", "location"
}

ENGINEER_KEYWORDS = {
    "fix", "repair", "path", "execute", "plan", "tool", "strategy", "time",
    "spiral", "raster", "sanding", "welding", "spraying", "ik", "kinematics",
    "trajectory", "speed", "velocity", "approach"
}


def route_message(message: str, has_defects: bool = False, has_plans: bool = False) -> AgentType:
    """
    Route a user message to the appropriate agent.
    
    Args:
        message: User's message text
        has_defects: Whether defects have been detected
        has_plans: Whether repair plans exist
        
    Returns:
        AgentType to handle the message
    """
    words = set(message.lower().split())
    
    # Check for engineer keywords first (more specific)
    if words & ENGINEER_KEYWORDS:
        return AgentType.ENGINEER
    
    # Check for inspector keywords
    if words & INSPECTOR_KEYWORDS:
        return AgentType.INSPECTOR
    
    # Default to supervisor for general queries
    return AgentType.SUPERVISOR


# ============ AGENT IMPLEMENTATIONS ============

class SupervisorAgent:
    """
    The Router & Front Desk.
    
    Handles greetings, help, and general questions.
    Provides context about the system and current status.
    """
    
    def __init__(self):
        self.agent_type = AgentType.SUPERVISOR
    
    def process(self, message: str, state: TeamState) -> AgentMessage:
        """Process a message and generate response."""
        msg_lower = message.lower()
        defects = state.get("defects", [])
        plans = state.get("plans", [])
        
        # Greeting
        if any(word in msg_lower for word in ["hello", "hi", "hey", "good"]):
            return AgentMessage(
                agent=self.agent_type,
                content=self._greeting_response(defects, plans)
            )
        
        # Help
        if "help" in msg_lower or "what can" in msg_lower:
            return AgentMessage(
                agent=self.agent_type,
                content=self._help_response()
            )
        
        # Status
        if "status" in msg_lower or "summary" in msg_lower:
            return AgentMessage(
                agent=self.agent_type,
                content=self._status_response(defects, plans)
            )
        
        # Default response
        return AgentMessage(
            agent=self.agent_type,
            content=self._default_response(defects)
        )
    
    def _greeting_response(self, defects: List, plans: List) -> str:
        if not defects:
            return (
                "👋 Welcome to the AARR Control Station!\n\n"
                "I'm the **Supervisor** coordinating our repair team.\n\n"
                "To get started:\n"
                "1. Load a part from the sidebar\n"
                "2. Click **Scan Part** to detect defects\n"
                "3. Ask me anything!"
            )
        else:
            return (
                f"👋 Hello! We currently have **{len(defects)} defect(s)** detected.\n\n"
                f"{'Plans are ready for approval.' if plans else 'Ready to generate repair plans.'}\n\n"
                "Ask the **Inspector** 👁️ about defects, or the **Engineer** 🔧 about repairs."
            )
    
    def _help_response(self) -> str:
        return (
            "🤖 **I'm part of a 3-agent team:**\n\n"
            "**👁️ Inspector** - Vision Expert\n"
            "- Ask: *'Show me defects'*, *'Where is the rust?'*\n"
            "- Controls the 3D viewer to highlight issues\n\n"
            "**🔧 Process Engineer** - Robotics Expert\n"
            "- Ask: *'Plan the repair'*, *'What tool to use?'*\n"
            "- Calculates paths and repair strategies\n\n"
            "**🤖 Supervisor (me)** - Coordinator\n"
            "- General questions and system status"
        )
    
    def _status_response(self, defects: List, plans: List) -> str:
        if not defects:
            return "📊 **Status**: No part loaded or scanned. Load a part to begin!"
        
        high = sum(1 for d in defects if d.get("severity") == "high")
        medium = sum(1 for d in defects if d.get("severity") == "medium")
        low = sum(1 for d in defects if d.get("severity") == "low")
        
        status = f"📊 **Status Summary**\n\n"
        status += f"**Defects**: {len(defects)} total\n"
        status += f"- 🔴 High: {high}\n"
        status += f"- 🟡 Medium: {medium}\n"
        status += f"- 🟢 Low: {low}\n\n"
        
        if plans:
            total_time = sum(p.get("estimated_time_seconds", 0) for p in plans)
            status += f"**Plans**: {len(plans)} repair plans ready\n"
            status += f"- Est. time: {total_time}s"
        else:
            status += "**Plans**: Not generated yet"
        
        return status
    
    def _default_response(self, defects: List) -> str:
        if defects:
            return (
                "I can help you with general questions. For specific tasks:\n"
                "- Say *'inspect'* or *'show defects'* → **Inspector** 👁️\n"
                "- Say *'repair'* or *'plan'* → **Engineer** 🔧"
            )
        return "Please load a part and scan it first. Then I can help you inspect and repair!"


class InspectorAgent:
    """
    The Vision Expert.
    
    Analyzes defect data and controls the 3D viewer.
    Can highlight defects, zoom to locations, show normals.
    """
    
    def __init__(self):
        self.agent_type = AgentType.INSPECTOR
    
    def process(self, message: str, state: TeamState) -> AgentMessage:
        """Process a message and generate response with UI commands."""
        msg_lower = message.lower()
        defects = state.get("defects", [])
        
        if not defects:
            return AgentMessage(
                agent=self.agent_type,
                content="👁️ No defects detected yet. Please scan the part first.",
                ui_commands=[]
            )
        
        ui_commands = []
        
        # Severity-based queries
        if any(word in msg_lower for word in ["high", "critical", "severe", "worst"]):
            return self._handle_severity_query(defects, "high")
        
        if any(word in msg_lower for word in ["medium", "moderate"]):
            return self._handle_severity_query(defects, "medium")
        
        if any(word in msg_lower for word in ["low", "minor"]):
            return self._handle_severity_query(defects, "low")
        
        # Type-based queries
        for dtype in ["crack", "rust", "corrosion", "pitting", "erosion", "wear", "damage"]:
            if dtype in msg_lower:
                return self._handle_type_query(defects, dtype)
        
        # Location-based queries
        location_map = {
            "top": lambda d: d["position"][2] > 0,
            "bottom": lambda d: d["position"][2] < 0,
            "left": lambda d: d["position"][1] > 0,
            "right": lambda d: d["position"][1] < 0,
            "front": lambda d: d["position"][0] > 0,
            "back": lambda d: d["position"][0] < 0,
        }
        
        for loc, filter_fn in location_map.items():
            if loc in msg_lower:
                return self._handle_location_query(defects, loc, filter_fn)
        
        # General inspection
        if any(word in msg_lower for word in ["all", "show", "inspect", "scan", "defect"]):
            return self._handle_overview(defects)
        
        # Default - show first defect
        return self._handle_overview(defects)
    
    def _handle_severity_query(self, defects: List, severity: str) -> AgentMessage:
        """Handle queries about specific severity levels."""
        matching = [d for d in defects if d.get("severity") == severity]
        
        if not matching:
            return AgentMessage(
                agent=self.agent_type,
                content=f"👁️ No **{severity}**-severity defects found.",
                ui_commands=[]
            )
        
        d = matching[0]
        pos = d["position"]
        
        content = (
            f"👁️ Found **{len(matching)} {severity}-severity** defect(s).\n\n"
            f"**Primary**: {d['type'].replace('_', ' ').title()}\n"
            f"- Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})\n"
            f"- Confidence: {d.get('confidence', 0.9)*100:.0f}%\n\n"
            f"*Zooming to location...*"
        )
        
        return AgentMessage(
            agent=self.agent_type,
            content=content,
            ui_commands=[
                UICommand(type=UICommandType.HIGHLIGHT_DEFECT, position=tuple(pos), defect_index=defects.index(d)),
                UICommand(type=UICommandType.ZOOM_TO, position=tuple(pos))
            ]
        )
    
    def _handle_type_query(self, defects: List, dtype: str) -> AgentMessage:
        """Handle queries about specific defect types."""
        matching = [d for d in defects if dtype in d.get("type", "").lower()]
        
        if not matching:
            return AgentMessage(
                agent=self.agent_type,
                content=f"👁️ No **{dtype}** defects detected on this part.",
                ui_commands=[]
            )
        
        d = matching[0]
        pos = d["position"]
        
        content = (
            f"👁️ Found **{len(matching)} {dtype}** defect(s).\n\n"
            f"**Location**: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})\n"
            f"**Severity**: {d.get('severity', 'unknown').title()}\n"
            f"**Confidence**: {d.get('confidence', 0.9)*100:.0f}%\n\n"
            f"*Highlighting on 3D view...*"
        )
        
        return AgentMessage(
            agent=self.agent_type,
            content=content,
            ui_commands=[
                UICommand(type=UICommandType.HIGHLIGHT_DEFECT, position=tuple(pos), defect_index=defects.index(d)),
                UICommand(type=UICommandType.ZOOM_TO, position=tuple(pos))
            ]
        )
    
    def _handle_location_query(self, defects: List, location: str, filter_fn) -> AgentMessage:
        """Handle queries about specific locations."""
        matching = [d for d in defects if filter_fn(d)]
        
        if not matching:
            return AgentMessage(
                agent=self.agent_type,
                content=f"👁️ No defects in the **{location}** region.",
                ui_commands=[]
            )
        
        d = matching[0]
        pos = d["position"]
        
        content = (
            f"👁️ Inspecting **{location}** region...\n\n"
            f"**Found**: {d['type'].replace('_', ' ').title()}\n"
            f"**Severity**: {d.get('severity', 'unknown').title()}\n"
            f"**Position**: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})\n\n"
            f"*Camera moving to location...*"
        )
        
        return AgentMessage(
            agent=self.agent_type,
            content=content,
            ui_commands=[
                UICommand(type=UICommandType.HIGHLIGHT_DEFECT, position=tuple(pos)),
                UICommand(type=UICommandType.ZOOM_TO, position=tuple(pos))
            ]
        )
    
    def _handle_overview(self, defects: List) -> AgentMessage:
        """Handle general inspection overview."""
        content = f"👁️ **Inspection Report**: {len(defects)} defect(s) detected\n\n"
        
        for i, d in enumerate(defects):
            severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(d.get("severity"), "⚪")
            content += f"{i+1}. {severity_icon} **{d['type'].replace('_', ' ').title()}** - {d.get('severity', 'unknown')}\n"
        
        content += "\n*Ask about specific defects or locations for details.*"
        
        return AgentMessage(
            agent=self.agent_type,
            content=content,
            ui_commands=[UICommand(type=UICommandType.RESET_VIEW)]
        )


class ProcessEngineerAgent:
    """
    The Robotics Expert.
    
    Plans repair strategies, selects tools, calculates paths.
    Knows about IK, velocities, and efficiency.
    """
    
    def __init__(self):
        self.agent_type = AgentType.ENGINEER
    
    def process(self, message: str, state: TeamState) -> AgentMessage:
        """Process a message and generate response."""
        msg_lower = message.lower()
        defects = state.get("defects", [])
        plans = state.get("plans", [])
        
        if not defects:
            return AgentMessage(
                agent=self.agent_type,
                content="🔧 No defects to repair. Please scan the part first.",
                ui_commands=[]
            )
        
        # Planning queries
        if any(word in msg_lower for word in ["plan", "strategy", "approach"]):
            return self._handle_planning(defects, plans)
        
        # Tool queries
        if any(word in msg_lower for word in ["tool", "sanding", "welding", "spraying"]):
            return self._handle_tool_query(defects, plans)
        
        # Time/efficiency queries
        if any(word in msg_lower for word in ["time", "long", "fast", "efficiency"]):
            return self._handle_time_query(defects, plans)
        
        # Execution queries
        if any(word in msg_lower for word in ["execute", "start", "run", "go"]):
            return self._handle_execution(defects, plans)
        
        # Path queries
        if any(word in msg_lower for word in ["path", "trajectory", "motion"]):
            return self._handle_path_query(defects, plans)
        
        # Default - general repair info
        return self._handle_general(defects, plans)
    
    def _handle_planning(self, defects: List, plans: List) -> AgentMessage:
        """Handle repair planning queries."""
        if plans:
            content = "🔧 **Repair Plans Ready**\n\n"
            total_time = 0
            
            for i, plan in enumerate(plans):
                content += f"**{i+1}. {plan.get('defect_type', 'Unknown').title()}**\n"
                content += f"   Strategy: {plan.get('strategy', 'spiral').title()}\n"
                content += f"   Tool: {plan.get('tool', 'sanding_disc')}\n"
                time_s = plan.get('estimated_time_seconds', 30)
                total_time += time_s
                content += f"   Time: ~{time_s}s\n\n"
            
            content += f"**Total estimated time**: {total_time}s\n"
            content += "\n*Ready for approval. Click 'Approve' to proceed.*"
        else:
            # Generate plans on the fly
            content = "🔧 **Generating Repair Plan...**\n\n"
            total_time = 0
            
            for i, defect in enumerate(defects):
                plan = get_fallback_plan(defect.get("type", "unknown"))
                content += f"**{i+1}. {defect.get('type', 'Unknown').replace('_', ' ').title()}**\n"
                content += f"   Strategy: {plan['strategy'].title()}\n"
                content += f"   Tool: {plan['tool']}\n"
                time_s = plan['estimated_time_seconds']
                total_time += time_s
                content += f"   Time: ~{time_s}s\n\n"
            
            content += f"**Total estimated time**: {total_time}s\n"
            content += "\n*Click 'Generate Plan' to finalize.*"
        
        return AgentMessage(
            agent=self.agent_type,
            content=content,
            ui_commands=[]
        )
    
    def _handle_tool_query(self, defects: List, plans: List) -> AgentMessage:
        """Handle tool selection queries."""
        content = "🔧 **Tool Recommendations**\n\n"
        
        tool_map = {
            "crack": ("filler_applicator", "Apply filler compound, then sand smooth"),
            "rust": ("sanding_disc_80", "80-grit for rust removal, then 120-grit finish"),
            "corrosion": ("chemical_treatment", "Apply rust converter, then protective coat"),
            "pitting": ("filler_applicator", "Fill pits, cure, then sand to profile"),
            "erosion": ("spray_gun", "Build up surface with spray deposition"),
            "wear": ("sanding_disc_120", "Light sanding then protective coating"),
        }
        
        for defect in defects:
            dtype = defect.get("type", "unknown").lower()
            for key, (tool, desc) in tool_map.items():
                if key in dtype:
                    content += f"**{dtype.replace('_', ' ').title()}**\n"
                    content += f"- Tool: `{tool}`\n"
                    content += f"- Process: {desc}\n\n"
                    break
            else:
                content += f"**{dtype.replace('_', ' ').title()}**: Standard sanding disc\n\n"
        
        return AgentMessage(
            agent=self.agent_type,
            content=content,
            ui_commands=[]
        )
    
    def _handle_time_query(self, defects: List, plans: List) -> AgentMessage:
        """Handle time/efficiency queries."""
        if plans:
            total_time = sum(p.get("estimated_time_seconds", 30) for p in plans)
        else:
            total_time = sum(get_fallback_plan(d.get("type", "unknown"))["estimated_time_seconds"] for d in defects)
        
        content = (
            f"🔧 **Time Estimate**\n\n"
            f"- **Defects**: {len(defects)}\n"
            f"- **Total repair time**: ~{total_time}s ({total_time/60:.1f} min)\n"
            f"- **Avg. per defect**: ~{total_time/len(defects):.0f}s\n\n"
            f"*Times include approach, repair, and verification.*"
        )
        
        return AgentMessage(
            agent=self.agent_type,
            content=content,
            ui_commands=[]
        )
    
    def _handle_execution(self, defects: List, plans: List) -> AgentMessage:
        """Handle execution queries."""
        if not plans:
            return AgentMessage(
                agent=self.agent_type,
                content="🔧 No approved plan yet. Generate and approve a plan first.",
                ui_commands=[]
            )
        
        content = (
            "🔧 **Ready for Execution**\n\n"
            f"- {len(plans)} repair operations queued\n"
            "- Robot will follow spiral toolpath\n"
            "- Max velocity: 0.1 m/s\n"
            "- Collision checking: enabled\n\n"
            "*Click 'Execute Repair' to begin. Toolpath will be shown on 3D view.*"
        )
        
        return AgentMessage(
            agent=self.agent_type,
            content=content,
            ui_commands=[UICommand(type=UICommandType.SHOW_TOOLPATH)]
        )
    
    def _handle_path_query(self, defects: List, plans: List) -> AgentMessage:
        """Handle path/trajectory queries."""
        content = (
            "🔧 **Toolpath Configuration**\n\n"
            "**Pattern**: Spiral (Archimedean)\n"
            "- Ensures complete coverage\n"
            "- Smooth velocity profile\n\n"
            "**Parameters**:\n"
            "- Start: 5mm above surface (approach)\n"
            "- Radius: Based on defect size\n"
            "- Loops: 2-3 (overlapping passes)\n"
            "- Tool orientation: Normal to surface\n\n"
            "*IK solved via damped least-squares.*"
        )
        
        return AgentMessage(
            agent=self.agent_type,
            content=content,
            ui_commands=[]
        )
    
    def _handle_general(self, defects: List, plans: List) -> AgentMessage:
        """Handle general repair queries."""
        high_count = sum(1 for d in defects if d.get("severity") == "high")
        
        content = (
            f"🔧 **Repair Overview**\n\n"
            f"- **Defects**: {len(defects)} ({high_count} high priority)\n"
            f"- **Plans**: {'Ready' if plans else 'Not generated'}\n\n"
            "I can help with:\n"
            "- `plan` - Generate repair strategy\n"
            "- `tool` - Tool recommendations\n"
            "- `time` - Time estimates\n"
            "- `path` - Toolpath details"
        )
        
        return AgentMessage(
            agent=self.agent_type,
            content=content,
            ui_commands=[]
        )


# ============ TEAM ORCHESTRATION ============

class MultiAgentTeam:
    """
    Orchestrates the multi-agent team.
    
    Routes messages to appropriate agents and manages shared state.
    """
    
    def __init__(self, defects: List[Dict] = None, plans: List[Dict] = None):
        """
        Initialize the team.
        
        Args:
            defects: List of detected defects
            plans: List of repair plans
        """
        self.supervisor = SupervisorAgent()
        self.inspector = InspectorAgent()
        self.engineer = ProcessEngineerAgent()
        
        self.state: TeamState = {
            "messages": [],
            "current_speaker": AgentType.SUPERVISOR,
            "defects": defects or [],
            "plans": plans or [],
            "ui_commands": []
        }
    
    def update_state(self, defects: List[Dict] = None, plans: List[Dict] = None):
        """Update the team's shared state."""
        if defects is not None:
            self.state["defects"] = defects
        if plans is not None:
            self.state["plans"] = plans
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a user message through the agent team.
        
        Args:
            message: User's message
            
        Returns:
            Dict with:
                - agent: Which agent responded
                - content: Response text
                - avatar: Agent's avatar emoji
                - ui_commands: List of UI commands
        """
        # Route to appropriate agent
        target_agent = route_message(
            message, 
            has_defects=bool(self.state["defects"]),
            has_plans=bool(self.state["plans"])
        )
        
        # Get agent instance
        agent_map = {
            AgentType.SUPERVISOR: self.supervisor,
            AgentType.INSPECTOR: self.inspector,
            AgentType.ENGINEER: self.engineer,
        }
        
        agent = agent_map[target_agent]
        
        # Process message
        response = agent.process(message, self.state)
        
        # Update state
        self.state["current_speaker"] = response.agent
        self.state["messages"].append({
            "role": "user",
            "content": message
        })
        self.state["messages"].append({
            "role": "assistant",
            "agent": response.agent.value,
            "content": response.content
        })
        
        # Convert UI commands to dicts
        ui_commands = []
        for cmd in response.ui_commands:
            ui_commands.append({
                "type": cmd.type.value,
                "position": cmd.position,
                "defect_index": cmd.defect_index,
                "data": cmd.data
            })
        
        return {
            "agent": response.agent.value,
            "agent_name": AGENT_NAMES[response.agent],
            "content": response.content,
            "avatar": AGENT_AVATARS[response.agent],
            "ui_commands": ui_commands
        }
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get the full chat history."""
        return self.state["messages"]


# Factory function
def create_team(defects: List[Dict] = None, plans: List[Dict] = None) -> MultiAgentTeam:
    """Create a new multi-agent team instance."""
    return MultiAgentTeam(defects=defects, plans=plans)
