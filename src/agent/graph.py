"""
LangGraph workflow for the repair agent.

Workflow: detect → classify → plan → [human approval] → execute → verify
"""

import asyncio
import os
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from enum import Enum
import operator
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

from src.agent.prompts import SYSTEM_PROMPT, CLASSIFICATION_PROMPT
from src.agent.tools import (
    DefectInfo, RepairPlan, get_fallback_plan, 
    parse_llm_response, validate_repair_plan
)
from src.config import config


class AgentState(TypedDict):
    """State for the repair agent workflow."""
    # Input
    defects: List[DefectInfo]
    robot_position: tuple
    
    # Processing
    current_defect_index: int
    repair_plans: List[RepairPlan]
    
    # Control
    approved: bool
    awaiting_approval: bool
    
    # Results
    completed_repairs: List[int]
    failed_repairs: List[int]
    
    # Messages
    messages: List[str]
    status: str


class RepairAgent:
    """
    LangGraph-based repair agent using Ollama or OpenAI.
    """
    
    def __init__(self, model: str = None):
        """
        Initialize the repair agent.
        
        Args:
            model: Model name (default from config)
        """
        agent_config = config.get("agent", {})
        self.provider = agent_config.get("provider", "ollama")
        self.model_name = model or agent_config.get("model", "qwen3:14b")
        self.timeout = agent_config.get("timeout", 30)
        self.max_retries = agent_config.get("max_retries", 3)
        
        # Initialize LLM based on provider
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=0.1,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            print(f"  Using OpenAI: {self.model_name}")
        else:
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=0.1,
            )
            print(f"  Using Ollama: {self.model_name}")
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("await_approval", self._await_approval_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("verify", self._verify_node)
        
        # Add edges
        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "plan")
        workflow.add_edge("plan", "await_approval")
        workflow.add_conditional_edges(
            "await_approval",
            self._approval_condition,
            {
                "approved": "execute",
                "rejected": END,
                "waiting": "await_approval",
            }
        )
        workflow.add_edge("execute", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._next_defect_condition,
            {
                "more_defects": "classify",
                "all_done": END,
            }
        )
        
        return workflow.compile()
    
    async def _classify_node(self, state: AgentState) -> AgentState:
        """Classify the current defect."""
        idx = state["current_defect_index"]
        defect = state["defects"][idx]
        
        state["messages"].append(f"Classifying defect {idx}: {defect.type}")
        state["status"] = f"Classifying {defect.type} defect..."
        
        # Build prompt
        prompt = CLASSIFICATION_PROMPT.format(
            defect_type=defect.type,
            x=defect.position[0],
            y=defect.position[1],
            z=defect.position[2],
            size=defect.size,
            confidence=defect.confidence,
        )
        
        # Call LLM with retry
        plan_dict = None
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.wait_for(
                    self._call_llm(prompt),
                    timeout=self.timeout
                )
                plan_dict = parse_llm_response(response)
                
                if plan_dict and validate_repair_plan(plan_dict):
                    break
                else:
                    state["messages"].append(f"  Retry {attempt + 1}: Invalid response")
                    plan_dict = None
                    
            except asyncio.TimeoutError:
                state["messages"].append(f"  Retry {attempt + 1}: Timeout")
            except Exception as e:
                state["messages"].append(f"  Retry {attempt + 1}: {str(e)}")
        
        # Use fallback if LLM failed
        if plan_dict is None:
            state["messages"].append("  Using fallback strategy")
            plan_dict = get_fallback_plan(defect.type)
        
        # Create repair plan
        repair_plan = RepairPlan(
            defect_index=idx,
            defect_type=defect.type,
            severity=plan_dict["severity"],
            strategy=plan_dict["strategy"],
            tool=plan_dict["tool"],
            estimated_time=plan_dict["estimated_time_seconds"],
            notes=plan_dict.get("notes", ""),
        )
        
        state["repair_plans"].append(repair_plan)
        state["messages"].append(
            f"  → {repair_plan.severity.upper()} severity, "
            f"{repair_plan.strategy} pattern, {repair_plan.tool}"
        )
        
        return state
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content
    
    def _plan_node(self, state: AgentState) -> AgentState:
        """Generate the repair plan summary."""
        idx = state["current_defect_index"]
        plan = state["repair_plans"][-1]
        
        state["messages"].append(
            f"Plan ready: {plan.strategy} with {plan.tool} "
            f"(~{plan.estimated_time}s)"
        )
        state["status"] = "Awaiting approval..."
        state["awaiting_approval"] = True
        
        return state
    
    def _await_approval_node(self, state: AgentState) -> AgentState:
        """Wait for human approval."""
        # This node is a checkpoint - actual approval happens externally
        return state
    
    def _approval_condition(self, state: AgentState) -> str:
        """Check approval status."""
        if state["approved"]:
            return "approved"
        elif not state["awaiting_approval"]:
            return "rejected"
        else:
            return "waiting"
    
    def _execute_node(self, state: AgentState) -> AgentState:
        """Execute the repair (placeholder - actual execution is external)."""
        idx = state["current_defect_index"]
        plan = state["repair_plans"][-1]
        
        state["messages"].append(f"Executing repair for defect {idx}...")
        state["status"] = f"Executing {plan.strategy} pattern..."
        state["awaiting_approval"] = False
        
        return state
    
    def _verify_node(self, state: AgentState) -> AgentState:
        """Verify the repair was successful."""
        idx = state["current_defect_index"]
        
        # Mark as completed (actual verification is external)
        state["completed_repairs"].append(idx)
        state["messages"].append(f"Defect {idx} repair verified ✓")
        
        # Move to next defect
        state["current_defect_index"] += 1
        state["approved"] = False
        
        return state
    
    def _next_defect_condition(self, state: AgentState) -> str:
        """Check if there are more defects to process."""
        if state["current_defect_index"] < len(state["defects"]):
            return "more_defects"
        else:
            state["status"] = "All repairs complete!"
            return "all_done"
    
    def create_initial_state(
        self,
        defects: List[DefectInfo],
        robot_position: tuple = (0, 0, 0.5)
    ) -> AgentState:
        """Create initial state for the workflow."""
        return AgentState(
            defects=defects,
            robot_position=robot_position,
            current_defect_index=0,
            repair_plans=[],
            approved=False,
            awaiting_approval=False,
            completed_repairs=[],
            failed_repairs=[],
            messages=[],
            status="Starting...",
        )
    
    async def run_classification(
        self,
        defects: List[DefectInfo]
    ) -> List[RepairPlan]:
        """
        Run classification for all defects (without execution).
        
        Args:
            defects: List of detected defects
            
        Returns:
            List of repair plans
        """
        plans = []
        
        for defect in defects:
            state = self.create_initial_state([defect])
            state = await self._classify_node(state)
            if state["repair_plans"]:
                plans.append(state["repair_plans"][-1])
        
        return plans


def create_agent(model: str = None) -> RepairAgent:
    """Factory function to create a repair agent."""
    return RepairAgent(model=model)
