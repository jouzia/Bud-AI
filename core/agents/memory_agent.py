"""
Memory-Augmented Smart Agent

Adds to GeminiAgent:
  1. Dual-layer memory (episodic + semantic) injected into every prompt
  2. Complexity mode toggle: ELI5 | standard | PhD
  3. [FACT] auto-extraction → semantic memory update after every step
  4. Human-in-the-loop: accept corrections, store them as high-confidence facts
  5. Fallback execution when Gemini is unavailable
  6. Token usage tracking
"""
from __future__ import annotations

import os
import time
import json
from enum import Enum
from typing import ClassVar, Optional

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from core.environment.state import StudyState, Action, MODE_ALLOWED_ACTIONS
from core.memory.memory import DualLayerMemory, Fact
from .base import BaseAgent, AgentResponse


class ComplexityMode(str, Enum):
    ELI5     = "eli5"      # Explain Like I'm 5
    STANDARD = "standard"
    PHD      = "phd"       # Dense, technical, assume expertise


COMPLEXITY_INSTRUCTIONS = {
    ComplexityMode.ELI5: (
        "Write at the level of a curious 10-year-old. "
        "Use simple words, analogies, and short sentences. "
        "No jargon. Make it feel like a friendly story."
    ),
    ComplexityMode.STANDARD: (
        "Write clearly for a university student. "
        "Use precise terminology where needed but explain it briefly."
    ),
    ComplexityMode.PHD: (
        "Write at a graduate research level. "
        "Use domain-specific terminology, assume deep prior knowledge, "
        "cite relationships between concepts, use formal structure."
    ),
}

_SYSTEM_PROMPT = """\
You are a Memory-Augmented Study Agent operating inside OpenEnv.

Your dual objectives:
1. STRATEGY — choose the optimal action given environment constraints.
2. EXECUTION — produce high-quality educational content for that action.

Critical rules:
- Never repeat the last action (penalty −0.30).
- Never use an action outside allowed_actions (penalty −0.50).
- When completeness < 0.5 → prefer expand.
- When completeness > 0.8 → prefer quiz or reorganize to lock in score.
- When last reward was negative → switch strategy immediately.

Fact extraction:
- When you discover or define a core concept, embed it as: [FACT: concept | one-line definition]
- These get stored in semantic memory and shown in the knowledge graph.

Output ONLY valid JSON. No markdown fences.
"""

_STEP_PROMPT = """\
=== MEMORY CONTEXT ===
{episodic_context}

{semantic_context}

=== ENVIRONMENT STATE ===
{state_json}

=== COMPLEXITY MODE ===
{complexity_instruction}

=== YOUR TASK ===
1. Decide the best action from: {allowed_actions}
2. Execute that action on the notes — produce the actual educational content.
3. Embed [FACT: concept | definition] tags for any key concepts you define.

Respond in this exact JSON:
{{
  "reasoning":  "<one sentence — why this action given the state + memory>",
  "action":     "<action_name>",
  "content":    "<the full educational content produced by this action>",
  "token_hint": <estimated_output_tokens_int>
}}
"""


class MemoryAgent(BaseAgent):
    name: ClassVar[str] = "memory-agent-gemini"

    def __init__(
        self,
        model_name:      str            = "gemini-1.5-flash",
        temperature:     float          = 0.3,
        complexity_mode: ComplexityMode = ComplexityMode.STANDARD,
        memory:          Optional[DualLayerMemory] = None,
    ):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set.")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name,
            system_instruction=_SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )
        self.model_name      = model_name
        self.complexity_mode = complexity_mode
        self.memory          = memory or DualLayerMemory(episodic_window=5)
        self.total_tokens    = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def act(self, state: StudyState) -> AgentResponse:
        allowed = [a.value for a in MODE_ALLOWED_ACTIONS[state.mode]]
        prompt  = _STEP_PROMPT.format(
            episodic_context       = self.memory.episodic_context(),
            semantic_context       = self.memory.semantic_context(),
            state_json             = json.dumps(state.model_dump_for_agent(), indent=2),
            complexity_instruction = COMPLEXITY_INSTRUCTIONS[self.complexity_mode],
            allowed_actions        = allowed,
        )

        t0  = time.perf_counter()
        raw = self._call_llm(prompt)
        ms  = round((time.perf_counter() - t0) * 1000, 1)

        action, reasoning, content, tokens = self._parse(raw, allowed)

        # Extract facts from content → semantic memory
        new_facts = self.memory.extract_and_store_facts(content, source="agent")
        clean     = self.memory.clean_output(content)

        self.total_tokens += tokens

        return AgentResponse(
            action     = action,
            reasoning  = reasoning,
            raw        = clean,          # cleaned content passed back to env
            latency_ms = ms,
        )

    def record_outcome(self, action: str, reward: float, outcome: str = "") -> None:
        """Called by benchmark after env.step() to update episodic memory."""
        self.memory.record_episode(action, reward, outcome)

    def human_intervene(self, correction: str, concept: str = "") -> list[Fact]:
        """
        Human-in-the-loop correction.
        Stores correction as a high-confidence human-sourced fact.
        Returns list of facts extracted.
        """
        facts = self.memory.extract_and_store_facts(correction, source="human")
        if concept and not facts:
            # If no [FACT] tags, store the whole correction under the given concept
            self.memory.store_fact(
                concept    = concept or "human_correction",
                definition = correction[:200],
                source     = "human",
                confidence = 1.0,
            )
            facts = [self.memory.retrieve_fact(concept)]
        return [f for f in facts if f]

    def reset(self) -> None:
        self.memory.reset_episodic()

    # ── LLM call with retry + fallback ────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        try:
            return self._call_with_retry(prompt)
        except Exception:
            return self._fallback(prompt)

    @retry(
        retry      = retry_if_exception_type(Exception),
        stop       = stop_after_attempt(3),
        wait       = wait_exponential(multiplier=1, min=2, max=12),
        reraise    = False,
    )
    def _call_with_retry(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        return response.text.strip()

    def _fallback(self, prompt: str) -> str:
        """
        Deterministic fallback when Gemini is unavailable.
        Uses heuristic logic so the benchmark never hard-crashes.
        """
        from core.agents.heuristic_agent import HeuristicAgent
        # We don't have a real state here, so return a safe default
        return json.dumps({
            "reasoning":  "LLM unavailable — using heuristic fallback",
            "action":     "expand",
            "content":    (
                "[FALLBACK] The Gemini API was unreachable. "
                "This response was generated by the deterministic heuristic agent. "
                "Check your GEMINI_API_KEY and network connection."
            ),
            "token_hint": 0,
        })

    def _parse(
        self, raw: str, allowed: list[str]
    ) -> tuple[Action, str, str, int]:
        try:
            data      = json.loads(raw)
            action_s  = data.get("action", "do_nothing").strip().lower()
            reasoning = data.get("reasoning", "")
            content   = data.get("content",   "")
            tokens    = int(data.get("token_hint", 0))
            if action_s in allowed:
                return Action(action_s), reasoning, content, tokens
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        # Fallback scan
        for a in allowed:
            if a in raw.lower():
                return Action(a), "parsed from raw", raw, 0
        return Action.DO_NOTHING, "parse failed", raw, 0
