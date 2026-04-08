"""
State model for the OpenEnv Study Assistant.
Immutable snapshot of the environment at each timestep.
Using Pydantic v2 for validation and serialisation.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, computed_field, model_validator


class Mode(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class Action(str, Enum):
    EXPAND     = "expand"
    SUMMARIZE  = "summarize"
    QUIZ       = "quiz"
    REORGANIZE = "reorganize"
    DO_NOTHING = "do_nothing"


# Which actions each mode permits
MODE_ALLOWED_ACTIONS: dict[Mode, frozenset[Action]] = {
    Mode.EASY:   frozenset({Action.EXPAND, Action.DO_NOTHING}),
    Mode.MEDIUM: frozenset({Action.EXPAND, Action.SUMMARIZE, Action.DO_NOTHING}),
    Mode.HARD:   frozenset(Action),
}

MODE_STEP_LIMITS: dict[Mode, int] = {
    Mode.EASY:   10,
    Mode.MEDIUM: 8,
    Mode.HARD:   5,
}


class StudyState(BaseModel):
    """
    Immutable snapshot of the study environment at a single timestep.
    Never mutate — always produce a new instance via env.step().
    """
    model_config = {"frozen": True}

    # Core fields
    mode:           Mode
    notes:          str                    = Field(default="",  description="Current knowledge base content")
    completeness:   Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    diversity_score:Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    steps_left:     Annotated[int,   Field(ge=0)]           = 0
    action_history: tuple[Action, ...] = Field(default_factory=tuple)
    penalty:        float = 0.0
    timestamp:      datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def steps_used(self) -> int:
        return len(self.action_history)

    @computed_field
    @property
    def allowed_actions(self) -> list[str]:
        return [a.value for a in MODE_ALLOWED_ACTIONS[self.mode]]

    @computed_field
    @property
    def last_action(self) -> str | None:
        return self.action_history[-1].value if self.action_history else None

    @computed_field
    @property
    def is_terminal(self) -> bool:
        return self.steps_left == 0 or self.completeness >= 1.0

    def model_dump_for_agent(self) -> dict:
        """Minimal dict sent to the LLM — no noise, just signal."""
        return {
            "mode":           self.mode.value,
            "notes_preview":  self.notes[:300] + "..." if len(self.notes) > 300 else self.notes,
            "completeness":   round(self.completeness, 3),
            "diversity_score":round(self.diversity_score, 3),
            "steps_left":     self.steps_left,
            "action_history": [a.value for a in self.action_history],
            "allowed_actions":self.allowed_actions,
            "penalty_so_far": round(self.penalty, 3),
        }
