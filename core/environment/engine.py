"""
OpenEnv Study Environment Engine.

Design principles:
- Pure functions where possible (step() returns new state, never mutates)
- All reward logic isolated in _compute_reward()
- Mode constraints enforced at the engine level, not the agent level
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .state import (
    Action, Mode, StudyState,
    MODE_ALLOWED_ACTIONS, MODE_STEP_LIMITS,
)


# ---------------------------------------------------------------------------
# Reward weights — tweak these to tune difficulty
# ---------------------------------------------------------------------------
COMPLETENESS_WEIGHT = 0.50
DIVERSITY_WEIGHT    = 0.30
EFFICIENCY_WEIGHT   = 0.20
REPEAT_PENALTY      = 0.30
ILLEGAL_PENALTY     = 0.50
EMPTY_OP_PENALTY    = 0.20


@dataclass(frozen=True)
class StepResult:
    state:  StudyState
    reward: float
    done:   bool
    info:   dict


class StudyEnv:
    """
    Stateful wrapper around immutable StudyState.

    Usage:
        env   = StudyEnv(mode=Mode.HARD)
        state = env.reset()
        while not state.is_terminal:
            result = env.step(action)
            state  = result.state
    """

    def __init__(self, mode: Mode | str = Mode.HARD, seed: int | None = None):
        self.mode  = Mode(mode) if isinstance(mode, str) else mode
        self._rng  = random.Random(seed)
        self._state: StudyState | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> StudyState:
        self._state = StudyState(
            mode        = self.mode,
            notes       = self._initial_notes(),
            completeness= 0.1,
            steps_left  = MODE_STEP_LIMITS[self.mode],
        )
        return self._state

    def step(self, action: Action | str) -> StepResult:
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        action = Action(action) if isinstance(action, str) else action
        state  = self._state
        reward, info = self._compute_reward(state, action)
        new_state    = self._apply_action(state, action, reward)
        done         = new_state.is_terminal
        self._state  = new_state
        return StepResult(state=new_state, reward=reward, done=done, info=info)

    @property
    def current_state(self) -> StudyState | None:
        return self._state

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------

    def _compute_reward(self, state: StudyState, action: Action) -> tuple[float, dict]:
        info: dict = {"penalties": [], "bonuses": []}
        reward = 0.0

        # 1. Illegal action in this mode
        if action not in MODE_ALLOWED_ACTIONS[self.mode]:
            info["penalties"].append(f"illegal_action:{ILLEGAL_PENALTY}")
            return -ILLEGAL_PENALTY, info

        # 2. Penalty for repeating the last action
        if state.last_action == action.value:
            reward -= REPEAT_PENALTY
            info["penalties"].append(f"repeat:{REPEAT_PENALTY}")

        # 3. Penalty for operating on empty notes
        if not state.notes.strip() and action in (Action.SUMMARIZE, Action.QUIZ, Action.REORGANIZE):
            reward -= EMPTY_OP_PENALTY
            info["penalties"].append(f"empty_op:{EMPTY_OP_PENALTY}")

        # 4. Completeness delta reward
        delta_c = self._completeness_delta(action, state)
        reward  += delta_c * COMPLETENESS_WEIGHT
        if delta_c > 0:
            info["bonuses"].append(f"completeness_delta:{delta_c:.3f}")

        # 5. Diversity bonus
        unique_actions = len(set(state.action_history) | {action})
        diversity      = unique_actions / max(len(Action) - 1, 1)
        reward        += diversity * DIVERSITY_WEIGHT
        info["bonuses"].append(f"diversity:{diversity:.3f}")

        # 6. Efficiency bonus (rewarded for doing useful work with fewer steps)
        if state.steps_left <= 2 and state.completeness > 0.7:
            reward += 0.15 * EFFICIENCY_WEIGHT
            info["bonuses"].append("efficiency_bonus")

        return round(reward, 4), info

    def _completeness_delta(self, action: Action, state: StudyState) -> float:
        """How much each action moves the completeness needle."""
        deltas = {
            Action.EXPAND:     0.25 if state.completeness < 0.5  else 0.10,
            Action.SUMMARIZE:  0.15 if state.completeness > 0.3  else -0.05,
            Action.QUIZ:       0.20 if state.completeness > 0.5  else 0.05,
            Action.REORGANIZE: 0.15,
            Action.DO_NOTHING: 0.0,
        }
        return deltas.get(action, 0.0)

    def _apply_action(self, state: StudyState, action: Action, reward: float) -> StudyState:
        """Produce next immutable state."""
        delta_c     = max(0.0, self._completeness_delta(action, state))
        new_comp    = min(1.0, state.completeness + delta_c)
        new_notes   = self._update_notes(state.notes, action, new_comp)
        new_penalty = state.penalty + sum(
            float(p.split(":")[1]) for p in
            [x for x in [f"{reward:.4f}"] if reward < 0]  # simplified
        ) if reward < 0 else state.penalty
        unique      = len(set(state.action_history) | {action})
        diversity   = round(unique / max(len(Action) - 1, 1), 3)

        return state.model_copy(update={
            "notes":           new_notes,
            "completeness":    round(new_comp, 4),
            "diversity_score": diversity,
            "steps_left":      max(0, state.steps_left - 1),
            "action_history":  state.action_history + (action,),
            "penalty":         round(new_penalty, 4),
        })

    def _update_notes(self, notes: str, action: Action, completeness: float) -> str:
        templates = {
            Action.EXPAND: [
                "\n\n📖 [EXPANDED] Added detailed explanation covering key concepts, "
                f"definitions and examples. Completeness now at {completeness:.0%}.",
                "\n\n📖 [EXPANDED] Incorporated additional context, background theory "
                f"and worked examples. Completeness now at {completeness:.0%}.",
            ],
            Action.SUMMARIZE: [
                "\n\n✂️ [SUMMARIZED] Condensed verbose sections into concise bullet points. "
                f"Core ideas preserved. Completeness at {completeness:.0%}.",
            ],
            Action.QUIZ: [
                "\n\n❓ [QUIZ] Generated 3 practice questions + model answers to reinforce retention. "
                f"Completeness at {completeness:.0%}.",
            ],
            Action.REORGANIZE: [
                "\n\n🗂️ [REORGANIZED] Applied hierarchical structure: Overview → Details → Summary. "
                f"Completeness at {completeness:.0%}.",
            ],
            Action.DO_NOTHING: [""],
        }
        choices = templates.get(action, [""])
        return notes + self._rng.choice(choices)

    def _initial_notes(self) -> str:
        return (
            "# Study Session\n\n"
            "Topic: Machine Learning Fundamentals\n\n"
            "Initial content: Neural networks are computational models inspired by "
            "biological neural networks. They consist of layers of interconnected nodes."
        )
