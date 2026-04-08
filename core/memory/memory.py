"""
Dual-Layer Memory System

Episodic Memory  — short-term ring buffer of recent (action, reward) pairs.
                   Lets the agent avoid repeating mistakes within an episode.

Semantic Memory  — long-term key-value store of extracted [FACT] concepts.
                   Persists across episodes. Simulates a vector store without
                   the infra overhead. Swap self._store for ChromaDB/Pinecone
                   in production by just replacing _retrieve/_store calls.

Design: both layers are encapsulated here. Nothing outside this module
        touches the raw dicts — always go through the public API.
"""
from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Episode:
    action:    str
    reward:    float
    outcome:   str   = ""          # short description of what happened
    timestamp: str   = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Fact:
    concept:    str
    definition: str
    source:     str   = "agent"    # "agent" | "human" (human-in-the-loop corrections)
    confidence: float = 1.0
    created_at: str   = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Core memory class ─────────────────────────────────────────────────────────

class DualLayerMemory:
    """
    Thread-safe-ish memory for a single agent instance.

    Episodic: ring buffer of last `episodic_window` episodes.
    Semantic: dict keyed by lowercase concept name.
    """

    FACT_PATTERN = re.compile(
        r'\[FACT:\s*(?P<concept>[^|]+?)\s*\|\s*(?P<definition>[^\]]+?)\s*\]',
        re.IGNORECASE,
    )

    def __init__(self, episodic_window: int = 5, persist_path: Optional[Path] = None):
        self._episodic: deque[Episode]   = deque(maxlen=episodic_window)
        self._semantic: dict[str, Fact]  = {}
        self._persist_path               = persist_path
        if persist_path and persist_path.exists():
            self._load(persist_path)

    # ── Episodic ──────────────────────────────────────────────────────────────

    def record_episode(self, action: str, reward: float, outcome: str = "") -> None:
        self._episodic.appendleft(Episode(action=action, reward=reward, outcome=outcome))

    @property
    def recent_episodes(self) -> list[Episode]:
        return list(self._episodic)

    @property
    def last_reward(self) -> float:
        return self._episodic[0].reward if self._episodic else 1.0

    @property
    def last_action(self) -> Optional[str]:
        return self._episodic[0].action if self._episodic else None

    def was_penalised_recently(self, threshold: float = 0.1) -> bool:
        return bool(self._episodic) and self._episodic[0].reward < threshold

    def episodic_context(self) -> str:
        if not self._episodic:
            return "No previous actions this session."
        lines = [
            f"  [{e.action.upper():<12}] reward={e.reward:+.2f}"
            + (f" — {e.outcome}" if e.outcome else "")
            for e in self._episodic
        ]
        return "Recent actions (newest first):\n" + "\n".join(lines)

    # ── Semantic ──────────────────────────────────────────────────────────────

    def store_fact(self, concept: str, definition: str,
                   source: str = "agent", confidence: float = 1.0) -> None:
        key = concept.strip().lower()
        # Human corrections override agent facts (higher confidence)
        if key in self._semantic and source == "agent":
            if self._semantic[key].source == "human":
                return  # never overwrite human corrections with agent guesses
        self._semantic[key] = Fact(
            concept    = concept.strip(),
            definition = definition.strip(),
            source     = source,
            confidence = confidence,
        )
        if self._persist_path:
            self._save(self._persist_path)

    def retrieve_fact(self, concept: str) -> Optional[Fact]:
        return self._semantic.get(concept.strip().lower())

    def all_facts(self) -> list[Fact]:
        return sorted(self._semantic.values(), key=lambda f: f.concept)

    def semantic_context(self, max_facts: int = 8) -> str:
        facts = self.all_facts()[:max_facts]
        if not facts:
            return "No learned concepts yet."
        lines = [f"  • {f.concept}: {f.definition[:80]}" for f in facts]
        return "Known concepts:\n" + "\n".join(lines)

    # ── Auto-extraction from LLM output ──────────────────────────────────────

    def extract_and_store_facts(self, text: str, source: str = "agent") -> list[Fact]:
        """
        Scan LLM output for [FACT: concept | definition] tags.
        Store each one. Return the list of newly extracted facts.
        """
        extracted = []
        for match in self.FACT_PATTERN.finditer(text):
            concept    = match.group("concept")
            definition = match.group("definition")
            self.store_fact(concept, definition, source=source)
            extracted.append(Fact(concept=concept, definition=definition, source=source))
        return extracted

    def clean_output(self, text: str) -> str:
        """Strip [FACT:...] tags from text before displaying to user."""
        return self.FACT_PATTERN.sub("", text).strip()

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset_episodic(self) -> None:
        """Call between episodes. Semantic memory persists."""
        self._episodic.clear()

    def reset_all(self) -> None:
        self._episodic.clear()
        self._semantic.clear()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, path: Path) -> None:
        try:
            data = {
                "semantic": {
                    k: {
                        "concept":    v.concept,
                        "definition": v.definition,
                        "source":     v.source,
                        "confidence": v.confidence,
                        "created_at": v.created_at,
                    }
                    for k, v in self._semantic.items()
                }
            }
            path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass  # persistence is best-effort

    def _load(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text())
            for k, v in data.get("semantic", {}).items():
                self._semantic[k] = Fact(**v)
        except Exception:
            pass
