from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    user_goal: str
    success_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    current_plan: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "read-only"
    iteration_count: int = 0
    max_iterations: int = 8
    is_done: bool = False
    last_step: Optional[str] = None
    last_status: Optional[str] = None
    session_id: Optional[str] = None

    def summary(self) -> str:
        return (
            f"goal={self.user_goal!r}, iterations={self.iteration_count}/{self.max_iterations}, "
            f"completed={len(self.completed_steps)}, failed={len(self.failed_steps)}, done={self.is_done}"
        )
