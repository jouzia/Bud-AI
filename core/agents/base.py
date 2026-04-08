"""
Agent base class. Every agent — LLM-backed or rule-based — implements this.
Enforces a consistent interface so the benchmark loop is agent-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from core.environment.state import StudyState, Action


@dataclass
class AgentResponse:
    action:     Action
    reasoning:  str  = ""
    raw:        str  = ""
    latency_ms: float = 0.0


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def act(self, state: StudyState) -> AgentResponse:
        """Given a state, return an action and reasoning."""
        ...

    def reset(self) -> None:
        """Called between episodes. Override if agent has episode-level state."""
        pass
