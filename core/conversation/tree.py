"""
Conversational Decision Tree Engine

Data-driven tree where each node has:
  - A question/prompt shown to the user
  - A set of predefined options (buttons)
  - An "other" escape hatch that opens a free-text input
  - A next_node map: option_key -> next node id
  - An optional action that fires when the node is reached

The UI layer renders this — the tree itself has zero Gradio imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class NodeType(str, Enum):
    QUESTION  = "question"   # presents options to user
    FREE_TEXT = "free_text"  # free input (reached via "other")
    ACTION    = "action"     # terminal — triggers something in the app
    INFO      = "info"       # just shows text, single "continue" button


@dataclass
class Option:
    key:   str           # internal identifier
    label: str           # what the user sees on the button
    icon:  str = ""      # emoji prefix


@dataclass
class TreeNode:
    id:          str
    type:        NodeType
    prompt:      str                          # question text shown to user
    options:     list[Option] = field(default_factory=list)
    next_node:   dict[str, str] = field(default_factory=dict)  # option.key -> node_id
    other_label: str  = "none of these — let me type"          # escape hatch label
    other_next:  str  = ""                                      # where "other" goes
    action_key:  str  = ""                                      # fires app action
    meta:        dict = field(default_factory=dict)             # arbitrary payload


@dataclass
class ConversationState:
    current_node_id:  str
    history:          list[tuple[str, str]] = field(default_factory=list)
    # history = [(prompt, user_choice), ...]
    collected:        dict[str, str] = field(default_factory=dict)
    # collected = {key: value} — builds up context as user progresses
    free_text_buffer: str = ""


class DecisionTree:
    """
    Holds the full node graph.
    Call .start() to get initial state.
    Call .advance(state, choice) to move forward.
    """

    def __init__(self, nodes: list[TreeNode], root_id: str):
        self._nodes:  dict[str, TreeNode] = {n.id: n for n in nodes}
        self._root_id = root_id

    def start(self) -> ConversationState:
        return ConversationState(current_node_id=self._root_id)

    def get_node(self, node_id: str) -> TreeNode:
        node = self._nodes.get(node_id)
        if not node:
            raise KeyError(f"Node '{node_id}' not found in tree")
        return node

    def current(self, state: ConversationState) -> TreeNode:
        return self.get_node(state.current_node_id)

    def advance(
        self,
        state:       ConversationState,
        choice_key:  str,
        free_text:   str = "",
    ) -> ConversationState:
        """
        Move the conversation forward one step.
        choice_key = option key OR "other" for free-text path.
        Returns NEW ConversationState (immutable pattern).
        """
        node    = self.current(state)
        history = list(state.history)
        collected = dict(state.collected)

        if choice_key == "other":
            collected["free_text"] = free_text
            history.append((node.prompt, f"[custom] {free_text}"))
            next_id = node.other_next or self._root_id
        else:
            opt = next((o for o in node.options if o.key == choice_key), None)
            label = opt.label if opt else choice_key
            history.append((node.prompt, label))
            collected[node.id] = choice_key
            next_id = node.next_node.get(choice_key, "")

        return ConversationState(
            current_node_id  = next_id,
            history          = history,
            collected        = collected,
            free_text_buffer = free_text if choice_key == "other" else "",
        )

    def is_terminal(self, state: ConversationState) -> bool:
        if not state.current_node_id:
            return True
        node = self._nodes.get(state.current_node_id)
        return node is None or node.type == NodeType.ACTION

    def build_context_summary(self, state: ConversationState) -> str:
        """Formats collected choices into a readable summary for the LLM prompt."""
        lines = ["User guided session context:"]
        for prompt, answer in state.history:
            lines.append(f"  Q: {prompt}")
            lines.append(f"  A: {answer}")
        if state.collected.get("free_text"):
            lines.append(f"  Custom input: {state.collected['free_text']}")
        return "\n".join(lines)
