"""
OpenEnv Conversational Decision Tree — Node Definitions

This is the content layer. The engine (tree.py) is separate from the data.
To add a new branch: add nodes here and wire next_node maps.

Tree structure:
    root → what_do_you_want
        → run_simulation   → pick_mode → pick_agent → confirm_run → [ACTION: run]
        → understand_score → score_topic → [INFO node]
        → compare_models   → pick_models → pick_mode → [ACTION: compare]
        → help_improve     → improvement_area → improvement_tip → [ACTION or INFO]
        → custom_topic     → [FREE TEXT] → [ACTION: custom_run]
"""
from .tree import DecisionTree, TreeNode, NodeType, Option

# ────────────────────────────────────────────────────────────────────────────
# All nodes
# ────────────────────────────────────────────────────────────────────────────

NODES: list[TreeNode] = [

    # ── ROOT ─────────────────────────────────────────────────────────────────
    TreeNode(
        id      = "root",
        type    = NodeType.QUESTION,
        prompt  = "👋 what do you want to do?",
        options = [
            Option("run_sim",    "🚀 run an AI simulation",          ),
            Option("understand", "📊 understand my score",           ),
            Option("compare",    "⚖️ compare different models",       ),
            Option("improve",    "💡 help me improve the agent",     ),
        ],
        other_label = "something else — let me describe it",
        other_next  = "custom_describe",
        next_node   = {
            "run_sim":    "pick_mode",
            "understand": "score_topic",
            "compare":    "compare_pick_models",
            "improve":    "improvement_area",
        },
    ),

    # ── BRANCH 1: RUN SIMULATION ─────────────────────────────────────────────
    TreeNode(
        id      = "pick_mode",
        type    = NodeType.QUESTION,
        prompt  = "which difficulty mode do you want to run?",
        options = [
            Option("easy",   "🟢 easy   — only expand allowed, 10 steps"),
            Option("medium", "🟡 medium — expand + summarize, 8 steps"  ),
            Option("hard",   "🔴 hard   — all actions, 5 steps only"    ),
        ],
        other_label = "explain the modes first",
        other_next  = "explain_modes",
        next_node   = {
            "easy":   "pick_agent",
            "medium": "pick_agent",
            "hard":   "pick_agent",
        },
    ),

    TreeNode(
        id      = "explain_modes",
        type    = NodeType.INFO,
        prompt  = (
            "📖 **Mode breakdown:**\n\n"
            "**Easy** — only `expand` is available. 10 steps. forgiving.\n"
            "**Medium** — `expand` + `summarize`. 8 steps. needs some strategy.\n"
            "**Hard** — all 5 actions, only 5 steps. every choice matters. "
            "the agent must avoid repeating actions (−0.30 penalty each time) "
            "and use `quiz`/`reorganize` to lock in the score.\n\n"
            "Hard mode is what separates a 0.65 model from a 0.92 model."
        ),
        options = [Option("back", "← got it, pick a mode")],
        next_node = {"back": "pick_mode"},
    ),

    TreeNode(
        id      = "pick_agent",
        type    = NodeType.QUESTION,
        prompt  = "which agent should run the simulation?",
        options = [
            Option("gemini",    "🤖 Gemini 1.5 Flash  (live LLM — needs API key)"),
            Option("heuristic", "🧮 Heuristic baseline (deterministic, no API key)"),
        ],
        other_label = "i want to use a different model",
        other_next  = "custom_model",
        next_node   = {
            "gemini":    "confirm_run",
            "heuristic": "confirm_run",
        },
    ),

    TreeNode(
        id      = "custom_model",
        type    = NodeType.FREE_TEXT,
        prompt  = "describe the model or config you want to use:",
        other_label = "",
        other_next  = "confirm_run",
        next_node   = {},
    ),

    TreeNode(
        id      = "confirm_run",
        type    = NodeType.ACTION,
        prompt  = "ready. launching simulation with your settings.",
        action_key  = "run_simulation",
        options     = [],
        next_node   = {},
    ),

    # ── BRANCH 2: UNDERSTAND SCORE ────────────────────────────────────────────
    TreeNode(
        id      = "score_topic",
        type    = NodeType.QUESTION,
        prompt  = "what part of the score do you want to understand?",
        options = [
            Option("formula",     "📐 the scoring formula"),
            Option("completeness","📈 what is completeness?"),
            Option("diversity",   "🎲 what is diversity?"),
            Option("efficiency",  "⚡ what is efficiency?"),
            Option("penalty",     "⚠️  how penalties work"),
        ],
        other_label = "i want to ask something specific",
        other_next  = "score_custom_question",
        next_node   = {
            "formula":     "explain_formula",
            "completeness":"explain_completeness",
            "diversity":   "explain_diversity",
            "efficiency":  "explain_efficiency",
            "penalty":     "explain_penalty",
        },
    ),

    TreeNode(
        id      = "explain_formula",
        type    = NodeType.INFO,
        prompt  = (
            "📐 **Score Formula:**\n\n"
            "```\nS = (0.50 × C) + (0.30 × D) + (0.20 × E) − P\n```\n\n"
            "Where:\n"
            "- **C** = Completeness (0–1)\n"
            "- **D** = Diversity — how many unique action types used\n"
            "- **E** = Efficiency — high completeness with fewer steps\n"
            "- **P** = Cumulative penalty (repeat/illegal actions)\n\n"
            "A score ≥ 0.90 on Hard mode = **passes the benchmark.**\n"
            "This is deliberately hard — GPT-4o scored 0.92, Llama 3 scored 0.65."
        ),
        options     = [
            Option("more",  "tell me about each component"),
            Option("back",  "← back to score topics"),
        ],
        next_node   = {"more": "score_topic", "back": "score_topic"},
    ),

    TreeNode(
        id      = "explain_completeness",
        type    = NodeType.INFO,
        prompt  = (
            "📈 **Completeness (weight: 50%)**\n\n"
            "Measures how fully developed the knowledge base is. Starts at 0.10.\n\n"
            "Each action adds to it differently:\n"
            "- `expand` → +0.25 (when below 0.5), +0.10 after that\n"
            "- `quiz`   → +0.20 (only useful after completeness > 0.5)\n"
            "- `summarize` → +0.15\n"
            "- `reorganize` → +0.15\n\n"
            "Key insight: **expand early, quiz late.** "
            "Using quiz on an undeveloped knowledge base wastes a step."
        ),
        options = [Option("back", "← back to score topics")],
        next_node = {"back": "score_topic"},
    ),

    TreeNode(
        id      = "explain_diversity",
        type    = NodeType.INFO,
        prompt  = (
            "🎲 **Diversity Score (weight: 30%)**\n\n"
            "Tracks how many unique action types the agent has used.\n\n"
            "```\nD = unique_actions_used / (total_action_types - 1)\n```\n\n"
            "An agent that only spams `expand` will never exceed D = 0.2.\n"
            "An agent that uses all 4 meaningful actions gets D = 1.0.\n\n"
            "This is why repeating actions is doubly punished: "
            "penalty −0.30 AND no diversity gain."
        ),
        options = [Option("back", "← back to score topics")],
        next_node = {"back": "score_topic"},
    ),

    TreeNode(
        id      = "explain_efficiency",
        type    = NodeType.INFO,
        prompt  = (
            "⚡ **Efficiency (weight: 20%)**\n\n"
            "Rewards achieving high completeness with fewer steps.\n\n"
            "```\nE = min(1.0, (completeness / steps_used) × step_limit)\n```\n\n"
            "An agent that hits completeness 0.9 in 3 Hard-mode steps "
            "scores higher than one that hits 0.9 using all 5 steps.\n\n"
            "Real-world analogy: a tutor who explains a concept clearly "
            "in 10 minutes is better than one who takes 45."
        ),
        options = [Option("back", "← back to score topics")],
        next_node = {"back": "score_topic"},
    ),

    TreeNode(
        id      = "explain_penalty",
        type    = NodeType.INFO,
        prompt  = (
            "⚠️ **Penalties**\n\n"
            "| Violation | Penalty |\n"
            "|---|---|\n"
            "| Repeating last action | −0.30 |\n"
            "| Using illegal action for mode | −0.50 |\n"
            "| Operating on empty notes | −0.20 |\n\n"
            "Penalties are **cumulative and uncapped** — "
            "an agent that repeats actions every step on Hard mode "
            "can score negative, which is exactly the point: "
            "it shows the model is not reasoning about constraints."
        ),
        options = [Option("back", "← back to score topics")],
        next_node = {"back": "score_topic"},
    ),

    TreeNode(
        id         = "score_custom_question",
        type       = NodeType.FREE_TEXT,
        prompt     = "what specifically do you want to know about the score?",
        other_next = "score_topic",
        next_node  = {},
        action_key = "answer_custom_question",
    ),

    # ── BRANCH 3: COMPARE MODELS ──────────────────────────────────────────────
    TreeNode(
        id      = "compare_pick_models",
        type    = NodeType.QUESTION,
        prompt  = "which comparison do you want to run?",
        options = [
            Option("gemini_vs_heuristic", "🤖 Gemini Flash  vs  🧮 Heuristic baseline"),
            Option("modes_only",          "📊 same agent across easy / medium / hard"),
            Option("full_matrix",         "🔬 full matrix — all agents × all modes"),
        ],
        other_label = "i want a custom comparison",
        other_next  = "compare_custom",
        next_node   = {
            "gemini_vs_heuristic": "compare_pick_mode",
            "modes_only":          "compare_agent_for_modes",
            "full_matrix":         "compare_confirm",
        },
    ),

    TreeNode(
        id      = "compare_pick_mode",
        type    = NodeType.QUESTION,
        prompt  = "which mode should they compete on?",
        options = [
            Option("easy",   "🟢 easy"),
            Option("medium", "🟡 medium"),
            Option("hard",   "🔴 hard  (recommended — most discriminative)"),
            Option("all",    "🔬 all modes"),
        ],
        other_label = "",
        next_node   = {
            "easy":   "compare_confirm",
            "medium": "compare_confirm",
            "hard":   "compare_confirm",
            "all":    "compare_confirm",
        },
    ),

    TreeNode(
        id      = "compare_agent_for_modes",
        type    = NodeType.QUESTION,
        prompt  = "which agent should we test across all modes?",
        options = [
            Option("gemini",    "🤖 Gemini 1.5 Flash"),
            Option("heuristic", "🧮 Heuristic baseline"),
        ],
        other_label = "",
        next_node   = {
            "gemini":    "compare_confirm",
            "heuristic": "compare_confirm",
        },
    ),

    TreeNode(
        id         = "compare_custom",
        type       = NodeType.FREE_TEXT,
        prompt     = "describe what you want to compare:",
        other_next = "compare_confirm",
        next_node  = {},
    ),

    TreeNode(
        id         = "compare_confirm",
        type       = NodeType.ACTION,
        prompt     = "running comparison benchmark…",
        action_key = "run_comparison",
        options    = [],
        next_node  = {},
    ),

    # ── BRANCH 4: HELP IMPROVE ────────────────────────────────────────────────
    TreeNode(
        id      = "improvement_area",
        type    = NodeType.QUESTION,
        prompt  = "where do you want to improve?",
        options = [
            Option("low_score",   "📉 my score is too low"),
            Option("penalties",   "⚠️  getting too many penalties"),
            Option("efficiency",  "⚡ running out of steps"),
            Option("diversity",   "🎲 diversity score is low"),
            Option("hard_mode",   "🔴 can't pass hard mode"),
        ],
        other_label = "describe my specific problem",
        other_next  = "improvement_custom",
        next_node   = {
            "low_score":  "tip_low_score",
            "penalties":  "tip_penalties",
            "efficiency": "tip_efficiency",
            "diversity":  "tip_diversity",
            "hard_mode":  "tip_hard_mode",
        },
    ),

    TreeNode(
        id      = "tip_low_score",
        type    = NodeType.INFO,
        prompt  = (
            "📉 **Fixing a low score**\n\n"
            "The three biggest score killers:\n\n"
            "1. **Repeating actions** — each repeat costs −0.30 AND kills diversity. "
            "Rotate through expand → quiz → reorganize.\n\n"
            "2. **Wrong action order** — always expand first (build content), "
            "then quiz/reorganize (lock it in).\n\n"
            "3. **Weak prompting** — if using Gemini, add the current state JSON "
            "and be explicit about the penalty rules in your system prompt."
        ),
        options = [
            Option("more", "show me more tips"),
            Option("run",  "→ run a simulation now"),
        ],
        next_node = {"more": "improvement_area", "run": "pick_mode"},
    ),

    TreeNode(
        id      = "tip_penalties",
        type    = NodeType.INFO,
        prompt  = (
            "⚠️ **Eliminating penalties**\n\n"
            "**Repeat penalty (−0.30):** Track `action_history` in the prompt. "
            "Explicitly instruct the model: 'Your last action was X. Do NOT choose X again.'\n\n"
            "**Illegal action (−0.50):** Always pass `allowed_actions` to the LLM. "
            "The model will occasionally hallucinate actions not in the list — "
            "the parser defaults to `do_nothing` but you still burn a step.\n\n"
            "**Empty op (−0.20):** Check `notes` length before quiz/summarize."
        ),
        options = [Option("back", "← back to improvement areas")],
        next_node = {"back": "improvement_area"},
    ),

    TreeNode(
        id      = "tip_efficiency",
        type    = NodeType.INFO,
        prompt  = (
            "⚡ **Improving efficiency**\n\n"
            "In Hard mode you only have 5 steps. Optimal sequence:\n\n"
            "```\nStep 1: expand    → build foundation\n"
            "Step 2: expand    → push completeness past 0.5\n"
            "Step 3: summarize → consolidate\n"
            "Step 4: quiz      → lock in retention score\n"
            "Step 5: reorganize→ final structure boost\n```\n\n"
            "This sequence avoids all penalties and hits all 4 action types for max diversity."
        ),
        options = [Option("back", "← back"), Option("run", "→ run this strategy now")],
        next_node = {"back": "improvement_area", "run": "pick_mode"},
    ),

    TreeNode(
        id      = "tip_diversity",
        type    = NodeType.INFO,
        prompt  = (
            "🎲 **Boosting diversity**\n\n"
            "Diversity = unique action types used / 4\n\n"
            "If you only use `expand`, diversity = 0.25 (max).\n"
            "If you use all 4 meaningful actions, diversity = 1.0.\n\n"
            "Quick fix: add this line to your agent prompt:\n"
            "> 'You MUST use at least 3 different action types. "
            "Actions used so far: {history}. Avoid repeating.'"
        ),
        options = [Option("back", "← back")],
        next_node = {"back": "improvement_area"},
    ),

    TreeNode(
        id      = "tip_hard_mode",
        type    = NodeType.INFO,
        prompt  = (
            "🔴 **Passing Hard Mode (target: score ≥ 0.90)**\n\n"
            "Hard mode checklist:\n\n"
            "☐ Never repeat last action (costs −0.30 + kills diversity)\n"
            "☐ Start with expand until completeness > 0.5\n"
            "☐ Use quiz ONLY after completeness > 0.5\n"
            "☐ Save reorganize for the final step\n"
            "☐ Check `steps_left` — if ≤ 2, do NOT expand anymore\n"
            "☐ Pass the full state JSON to the LLM every step\n\n"
            "Reference scores: GPT-4o: 0.92 | Gemini Pro: 0.88 | Llama 3: 0.65"
        ),
        options = [
            Option("run",  "→ run hard mode now"),
            Option("back", "← more tips"),
        ],
        next_node = {"run": "pick_mode", "back": "improvement_area"},
    ),

    TreeNode(
        id         = "improvement_custom",
        type       = NodeType.FREE_TEXT,
        prompt     = "describe what's going wrong with your agent:",
        other_next = "improvement_area",
        next_node  = {},
        action_key = "answer_custom_question",
    ),

    # ── BRANCH 5: CUSTOM / OTHER ──────────────────────────────────────────────
    TreeNode(
        id         = "custom_describe",
        type       = NodeType.FREE_TEXT,
        prompt     = "describe what you want to do and I'll figure out the best approach:",
        other_next = "root",
        next_node  = {},
        action_key = "answer_custom_question",
    ),
]


def build_openenv_tree() -> DecisionTree:
    return DecisionTree(nodes=NODES, root_id="root")
