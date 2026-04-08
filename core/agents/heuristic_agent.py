"""
Heuristic Baseline Agent

A hand-coded rule-based agent that plays optimally given the reward function.
Used as:
  1. A sanity check — LLM agents should beat this
  2. A fast fallback if the API is unavailable
  3. A baseline row in the comparative analysis table
"""
from __future__ import annotations

from core.environment.state import StudyState, Action, MODE_ALLOWED_ACTIONS
from .base import BaseAgent, AgentResponse


class HeuristicAgent(BaseAgent):
    name = "heuristic-baseline"

    def act(self, state: StudyState) -> AgentResponse:
        allowed  = MODE_ALLOWED_ACTIONS[state.mode]
        last     = state.last_action
        comp     = state.completeness
        history  = [a.value for a in state.action_history]
        steps    = state.steps_left

        action, reasoning = self._policy(allowed, last, comp, history, steps)
        return AgentResponse(action=action, reasoning=reasoning)

    def _policy(self, allowed, last, comp, history, steps):
        # 1. Never repeat last action
        avoid = {Action(last)} if last else set()

        # 2. Final steps — lock in with quiz/reorganize
        if steps <= 2:
            for a in [Action.QUIZ, Action.REORGANIZE]:
                if a in allowed and a not in avoid:
                    return a, "locking in score with high-value action in final steps"

        # 3. Build completeness if low
        if comp < 0.5 and Action.EXPAND in allowed and Action.EXPAND not in avoid:
            return Action.EXPAND, "completeness below 0.5 — expanding knowledge base"

        # 4. Diversify — pick least-used allowed action
        counts = {a: history.count(a.value) for a in allowed if a not in avoid}
        if counts:
            best = min(counts, key=counts.get)
            return best, f"diversifying — {best.value} least used so far"

        # 5. Ultimate fallback
        for a in allowed:
            if a not in avoid:
                return a, "fallback selection"
        return Action.DO_NOTHING, "no valid action available"
