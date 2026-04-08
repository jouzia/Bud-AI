"""
Gemini LLM Agent

Production-grade implementation:
- Exponential backoff via tenacity (handles rate limits)
- Structured prompt with chain-of-thought reasoning
- Response parsing with fallback to do_nothing
- Latency tracking per call
"""
from __future__ import annotations

import json
import os
import time
from typing import ClassVar

import google.generativeai as genai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from core.environment.state import StudyState, Action, MODE_ALLOWED_ACTIONS
from .base import BaseAgent, AgentResponse


_SYSTEM_PROMPT = """\
You are an Autonomous Research Agent operating inside OpenEnv — a rigorous AI benchmark \
for stateful, long-horizon study tasks.

Your objective is to maximise the Learning Score:
    Score = 0.5 × Completeness + 0.3 × Diversity + 0.2 × Efficiency − Penalty

Critical rules you MUST follow:
1. NEVER repeat the last action — penalty of −0.30 per violation.
2. NEVER use an action not in allowed_actions — penalty of −0.50.
3. In Hard mode with ≤ 2 steps left, prioritise quiz or reorganize to lock in score.
4. If completeness > 0.8, do NOT use expand — it yields diminishing returns.
5. If notes are empty, do NOT use summarize, quiz, or reorganize.

Respond ONLY in valid JSON. No markdown fences. No explanation outside the JSON.
"""

_ACTION_PROMPT = """\
Current environment state:
{state_json}

Analyse the state step by step, then choose your action.

Respond in this exact JSON format:
{{
  "reasoning": "<one concise sentence explaining your strategy>",
  "action": "<action_name>"
}}

Valid action_name values: {allowed}
"""


class GeminiAgent(BaseAgent):
    name: ClassVar[str] = "gemini-1.5-flash"

    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.2):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. "
                "Export it: export GEMINI_API_KEY='your-key'"
            )
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name,
            system_instruction=_SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=256,
                response_mime_type="application/json",
            ),
        )
        self.model_name = model_name

    def act(self, state: StudyState) -> AgentResponse:
        allowed = [a.value for a in MODE_ALLOWED_ACTIONS[state.mode]]
        prompt  = _ACTION_PROMPT.format(
            state_json=json.dumps(state.model_dump_for_agent(), indent=2),
            allowed=allowed,
        )
        t0  = time.perf_counter()
        raw = self._call_with_retry(prompt)
        ms  = round((time.perf_counter() - t0) * 1000, 1)

        action, reasoning = self._parse(raw, allowed)
        return AgentResponse(
            action     = action,
            reasoning  = reasoning,
            raw        = raw,
            latency_ms = ms,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=False,
    )
    def _call_with_retry(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        return response.text.strip()

    def _parse(self, raw: str, allowed: list[str]) -> tuple[Action, str]:
        try:
            data      = json.loads(raw)
            action_str = data.get("action", "do_nothing").strip().lower()
            reasoning  = data.get("reasoning", "")
            if action_str in allowed:
                return Action(action_str), reasoning
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: scan raw text for any valid action name
        for a in allowed:
            if a in raw.lower():
                return Action(a), "parsed from raw text"
        return Action.DO_NOTHING, "parse failed — defaulting to do_nothing"
