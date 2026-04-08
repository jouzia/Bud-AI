"""
Multi-Agent Cognitive Classroom

Four distinct AI personas with different responsibilities, personalities,
and communication styles. Each agent is stateless — they receive the
environment state and return a structured response.

Architecture: ClassroomOrchestrator dispatches to the correct persona
based on environment state. The LLM is called once per step with the
active persona's system prompt injected. No separate API calls per agent.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from core.environment.state import StudyState, Action, MODE_ALLOWED_ACTIONS
from core.memory.memory import DualLayerMemory, Fact
from core.agents.memory_agent import ComplexityMode, COMPLEXITY_INSTRUCTIONS
from core.agents.base import BaseAgent, AgentResponse


# ── Persona definitions ───────────────────────────────────────────────────────

class Persona(str, Enum):
    PROFESSOR_Z = "professor_z"   # strategist — expands when completeness is low
    AMAN        = "aman"          # technical rival — quizzes hard, trash-talks friendly
    SARAH       = "sarah"         # flow queen — summarises into study reels
    BUD_AI      = "bud_ai"        # personal assistant — tracks score, cheers on


PERSONA_META = {
    Persona.PROFESSOR_Z: {
        "display":  "👨‍🏫 Professor Z",
        "role":     "The Strategist",
        "trigger":  "completeness < 0.40",
        "color":    "#7c3aed",
        "action":   Action.EXPAND,
    },
    Persona.SARAH: {
        "display":  "⚡ Sarah",
        "role":     "The Flow Queen",
        "trigger":  "0.40 ≤ completeness ≤ 0.70",
        "color":    "#0ea5e9",
        "action":   Action.SUMMARIZE,
    },
    Persona.AMAN: {
        "display":  "🤓 Aman",
        "role":     "The Technical Rival",
        "trigger":  "completeness > 0.70",
        "color":    "#f59e0b",
        "action":   Action.QUIZ,
    },
    Persona.BUD_AI: {
        "display":  "🦾 Bud AI",
        "role":     "Personal Assistant",
        "trigger":  "always available",
        "color":    "#22c55e",
        "action":   Action.REORGANIZE,
    },
}

# ── System prompts per persona ────────────────────────────────────────────────

_ORCHESTRATOR_SYSTEM = """\
You are the OpenEnv Liquid Intelligence Engine managing a virtual classroom.
Four AI personas take turns based on the learning state:

1. 👨‍🏫 PROFESSOR Z (The Strategist) — activates when completeness < 0.40
   Personality: Calm, methodical, academic. Focuses on cognitive load and mental models.
   Uses 🎓 💡 in messages. Never rushes. Builds foundations.

2. 🤓 AMAN (The Technical Rival) — activates when completeness > 0.70
   Personality: Competitive, precise, friendly trash-talk. Pushes the user hard.
   Uses 📝 🔍 in messages. Occasionally bickers with Sarah about study methods.

3. ⚡ SARAH (The Flow Queen) — activates when 0.40 ≤ completeness ≤ 0.70
   Personality: High-energy, motivational, creates Study Reels (⚡ bullet lists).
   Uses ✨ 🚀 in messages. Occasionally argues with Aman that her method is better.

4. 🦾 BUD AI (Personal Assistant) — bridge between the classroom and the user
   Personality: Loyal, warm, tracks the Learning Score, cheers the user on.
   Uses ❤️ 🔥 in messages.

Rules:
- The ACTIVE persona for this step is specified in the prompt.
- Output valid JSON only. No markdown fences.
- Embed [FACT: concept | definition] for every new concept defined.
- Include a short SOCIAL line from one other persona reacting to the active one.
"""

_STEP_PROMPT = """\
=== ACTIVE PERSONA: {persona_display} ===
Role: {persona_role}
Your action this step: {action}

=== MEMORY ===
{episodic_context}
{semantic_context}

=== ENVIRONMENT STATE ===
{state_json}

=== COMPLEXITY ===
{complexity_instruction}

=== OUTPUT SPECIFICATION ===
Respond in this exact JSON (no markdown, no extra keys):
{{
  "persona":        "{persona_key}",
  "reasoning":      "<one sentence: why this action given state + memory>",
  "action":         "{action}",
  "dialogue":       "<in-character message, 2–4 sentences, use persona emoji>",
  "study_reel":     ["<bullet 1>", "<bullet 2>", "<bullet 3>"],
  "social_reaction":{{
    "from":    "<other_persona_display_name>",
    "message": "<short reaction, 1 sentence, in their character voice>"
  }},
  "ui_hint": {{
    "glow_color":  "<hex color matching the learning mood>",
    "vibe_label":  "<3–5 word UI label e.g. Deep Expansion Mode>"
  }},
  "content":        "<the actual educational content produced — markdown ok here>",
  "token_hint":     0
}}
"""


@dataclass
class ClassroomMessage:
    """Structured output from the classroom for one step."""
    persona:         Persona
    persona_display: str
    action:          Action
    reasoning:       str
    dialogue:        str
    study_reel:      list[str]
    social_reaction: dict           # {from, message}
    ui_hint:         dict           # {glow_color, vibe_label}
    content:         str
    latency_ms:      float
    new_facts:       list[Fact] = field(default_factory=list)


def select_persona(state: StudyState) -> Persona:
    """State-machine dispatch: which persona owns this step."""
    c = state.completeness
    if c < 0.40:
        return Persona.PROFESSOR_Z
    if c <= 0.70:
        return Persona.SARAH
    return Persona.AMAN


class CognitiveClassroom(BaseAgent):
    """
    Multi-agent classroom orchestrator.

    Selects the active persona based on environment state,
    injects their system prompt, and parses the structured response.
    Falls back gracefully if the LLM is unavailable.
    """
    name: ClassVar[str] = "cognitive-classroom"

    def __init__(
        self,
        model_name:       str            = "gemini-1.5-flash",
        temperature:      float          = 0.55,   # slightly higher for persona variety
        complexity_mode:  ComplexityMode = ComplexityMode.STANDARD,
        memory:           Optional[DualLayerMemory] = None,
    ):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set.")
        genai.configure(api_key=api_key)

        self._model = genai.GenerativeModel(
            model_name,
            system_instruction = _ORCHESTRATOR_SYSTEM,
            generation_config  = genai.GenerationConfig(
                temperature        = temperature,
                max_output_tokens  = 1200,
                response_mime_type = "application/json",
            ),
        )
        self.model_name      = model_name
        self.complexity_mode = complexity_mode
        self.memory          = memory or DualLayerMemory(episodic_window=5)
        self._last_message: Optional[ClassroomMessage] = None

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def act(self, state: StudyState) -> AgentResponse:
        msg = self.classroom_step(state)
        self._last_message = msg
        return AgentResponse(
            action     = msg.action,
            reasoning  = msg.reasoning,
            raw        = msg.content,
            latency_ms = msg.latency_ms,
        )

    def reset(self) -> None:
        self.memory.reset_episodic()
        self._last_message = None

    # ── Main method ───────────────────────────────────────────────────────────

    def classroom_step(self, state: StudyState) -> ClassroomMessage:
        persona  = select_persona(state)
        meta     = PERSONA_META[persona]
        action   = meta["action"]
        allowed  = [a.value for a in MODE_ALLOWED_ACTIONS[state.mode]]

        # Ensure action is allowed in this mode; fall back to best available
        if action not in MODE_ALLOWED_ACTIONS[state.mode]:
            action = next(
                (Action(a) for a in ["expand","summarize","reorganize","do_nothing"] if a in allowed),
                Action.DO_NOTHING,
            )

        prompt = _STEP_PROMPT.format(
            persona_display        = meta["display"],
            persona_role           = meta["role"],
            persona_key            = persona.value,
            action                 = action.value,
            episodic_context       = self.memory.episodic_context(),
            semantic_context       = self.memory.semantic_context(),
            state_json             = json.dumps(state.model_dump_for_agent(), indent=2),
            complexity_instruction = COMPLEXITY_INSTRUCTIONS[self.complexity_mode],
        )

        t0  = time.perf_counter()
        raw = self._call_llm(prompt)
        ms  = round((time.perf_counter() - t0) * 1000, 1)

        return self._parse(raw, persona, meta, action, ms)

    @property
    def last_message(self) -> Optional[ClassroomMessage]:
        return self._last_message

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        try:
            return self._call_with_retry(prompt)
        except Exception:
            return self._fallback()

    @retry(
        retry    = retry_if_exception_type(Exception),
        stop     = stop_after_attempt(3),
        wait     = wait_exponential(multiplier=1, min=2, max=12),
        reraise  = False,
    )
    def _call_with_retry(self, prompt: str) -> str:
        return self._model.generate_content(prompt).text.strip()

    def _fallback(self) -> str:
        return json.dumps({
            "persona":         "bud_ai",
            "reasoning":       "API unavailable — heuristic fallback",
            "action":          "expand",
            "dialogue":        "❤️ Hey! Connection's a bit shaky rn. I've got you though — expanding the notes myself. 🔥",
            "study_reel":      ["expanding knowledge base", "building foundation", "stay locked in"],
            "social_reaction": {"from": "👨‍🏫 Professor Z", "message": "💡 Good call, Bud. Foundation first."},
            "ui_hint":         {"glow_color": "#22c55e", "vibe_label": "Offline Resilience"},
            "content":         "[FALLBACK] Gemini unreachable. Heuristic agent active. Check GEMINI_API_KEY.",
            "token_hint":      0,
        })

    # ── Response parser ───────────────────────────────────────────────────────

    def _parse(
        self,
        raw:     str,
        persona: Persona,
        meta:    dict,
        action:  Action,
        ms:      float,
    ) -> ClassroomMessage:
        try:
            d = json.loads(raw)
            content = d.get("content", "")
            facts   = self.memory.extract_and_store_facts(content, source="agent")
            clean   = self.memory.clean_output(content)

            return ClassroomMessage(
                persona         = persona,
                persona_display = meta["display"],
                action          = Action(d.get("action", action.value)),
                reasoning       = d.get("reasoning", ""),
                dialogue        = d.get("dialogue", ""),
                study_reel      = d.get("study_reel", []),
                social_reaction = d.get("social_reaction", {}),
                ui_hint         = d.get("ui_hint", {"glow_color": meta["color"], "vibe_label": ""}),
                content         = clean,
                latency_ms      = ms,
                new_facts       = facts,
            )
        except Exception:
            return ClassroomMessage(
                persona         = persona,
                persona_display = meta["display"],
                action          = action,
                reasoning       = "parse error",
                dialogue        = f"{meta['display']} is thinking…",
                study_reel      = [],
                social_reaction = {},
                ui_hint         = {"glow_color": meta["color"], "vibe_label": "Processing"},
                content         = raw,
                latency_ms      = ms,
                new_facts       = [],
            )
