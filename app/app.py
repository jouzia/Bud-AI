"""
OpenEnv Study Intelligence — Gradio Application v3.0
"Liquid Cognitive Classroom"

Design philosophy:
  - Neumorphic Liquid Zen Garden aesthetic — dark mode, fluid glows
  - Multi-agent classroom: Professor Z / Aman / Sarah / Bud AI
  - Zero business logic in this file — all delegated to core.*
  - State via gr.State — no globals, no race conditions
"""
from __future__ import annotations

import os, sys, traceback, json
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.classroom       import CognitiveClassroom, PERSONA_META, Persona, select_persona, ClassroomMessage
from core.agents.heuristic_agent import HeuristicAgent
from core.agents.memory_agent    import MemoryAgent, ComplexityMode
from core.benchmark              import run_episode, run_comparative_benchmark, EpisodeResult
from core.config                 import Config
from core.conversation           import build_openenv_tree, ConversationState, NodeType
from core.environment.state      import Mode, StudyState
from core.grading.grader         import GradeReport
from core.memory.memory          import DualLayerMemory
from core.telemetry              import reset_session, get_logger
from core.visualisation          import build_graph

log = get_logger("openenv.app")
cfg = Config.load()
TREE = build_openenv_tree()

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Liquid Zen Garden Neumorphic Dark Theme
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&family=Syne:wght@600;700;800&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: #07070f !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
    color: #e2e8f0 !important;
    overflow-x: hidden;
}

/* ── Liquid ambient background ── */
.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 15% 20%, rgba(124,58,237,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 85% 75%, rgba(14,165,233,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 50% at 50% 50%, rgba(34,197,94,0.03) 0%, transparent 70%);
    animation: liquid-drift 30s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
@keyframes liquid-drift {
    0%   { transform: scale(1)    rotate(0deg); }
    50%  { transform: scale(1.05) rotate(3deg) translate(1%, 1%); }
    100% { transform: scale(1.02) rotate(-2deg) translate(-1%, 2%); }
}

/* ── Neumorphic card surfaces ── */
.gr-box, .block {
    background: #111120 !important;
    border: 1px solid rgba(124,58,237,0.12) !important;
    border-radius: 20px !important;
    box-shadow:
        8px 8px 20px rgba(0,0,0,0.5),
        -4px -4px 12px rgba(255,255,255,0.02) !important;
    position: relative;
    z-index: 1;
}

/* ── Labels ── */
.label-wrap > span {
    color: #7c3aed !important;
    font-weight: 500 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: .08em !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Inputs ── */
textarea, input[type=text] {
    background: #0d0d1e !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(124,58,237,0.2) !important;
    border-radius: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    transition: border-color .2s, box-shadow .2s !important;
}
textarea:focus, input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.18), 0 0 20px rgba(124,58,237,0.1) !important;
    outline: none !important;
}

/* ── Choice buttons (conversation tree) ── */
.btn-choice {
    background: #111120 !important;
    border: 1px solid rgba(124,58,237,0.18) !important;
    color: #94a3b8 !important;
    border-radius: 14px !important;
    text-align: left !important;
    padding: 11px 16px !important;
    font-size: 13px !important;
    font-family: 'DM Sans', sans-serif !important;
    cursor: pointer !important;
    width: 100% !important;
    margin-bottom: 6px !important;
    transition: all .18s cubic-bezier(.175,.885,.32,1.275) !important;
    box-shadow: 4px 4px 10px rgba(0,0,0,0.4), -2px -2px 6px rgba(255,255,255,0.02) !important;
}
.btn-choice:hover {
    background: rgba(124,58,237,0.12) !important;
    border-color: #7c3aed !important;
    color: #c084fc !important;
    transform: translateX(4px) !important;
    box-shadow: 6px 6px 16px rgba(0,0,0,0.5), 0 0 12px rgba(124,58,237,0.2) !important;
}

/* ── Primary button ── */
.btn-primary {
    background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 50px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    cursor: pointer !important;
    transition: all .2s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4), 0 0 0 1px rgba(124,58,237,0.3) !important;
    letter-spacing: .02em !important;
}
.btn-primary:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 32px rgba(124,58,237,0.5), 0 0 0 1px rgba(124,58,237,0.5) !important;
}
.btn-primary:active { transform: scale(0.98) !important; }

/* ── Secondary button ── */
.btn-secondary {
    background: transparent !important;
    color: #64748b !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 50px !important;
    font-size: 12px !important;
    padding: 7px 16px !important;
    font-family: 'DM Sans', sans-serif !important;
    cursor: pointer !important;
    transition: all .15s !important;
}
.btn-secondary:hover {
    border-color: rgba(124,58,237,0.4) !important;
    color: #c084fc !important;
}

/* ── Tabs ── */
.tab-nav { border-bottom: 1px solid rgba(255,255,255,0.06) !important; }
.tab-nav button {
    color: #64748b !important;
    font-size: 13px !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 10px 18px !important;
    border-radius: 0 !important;
    transition: all .15s !important;
}
.tab-nav button.selected {
    color: #c084fc !important;
    border-bottom: 2px solid #7c3aed !important;
    background: transparent !important;
}

/* ── Markdown ── */
.markdown-body { color: #cbd5e1 !important; line-height: 1.7 !important; }
.markdown-body h1,.markdown-body h2,.markdown-body h3 {
    color: #c084fc !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
}
.markdown-body table th {
    background: rgba(124,58,237,0.15) !important;
    color: #c084fc !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.markdown-body table td { border-color: rgba(255,255,255,0.06) !important; }
.markdown-body strong { color: #e2e8f0 !important; }
.markdown-body code {
    background: rgba(124,58,237,0.15) !important;
    color: #c084fc !important;
    border-radius: 5px !important;
    padding: 2px 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}

/* ── Chat bubble styles ── */
.chat-wrap { display: flex; flex-direction: column; gap: 14px; padding: 4px 0; }
.chat-bubble {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    animation: bubble-in .3s cubic-bezier(.34,1.56,.64,1);
}
@keyframes bubble-in {
    from { opacity:0; transform: translateY(10px) scale(.97); }
    to   { opacity:1; transform: translateY(0) scale(1); }
}
.chat-avatar {
    width: 38px; height: 38px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
    box-shadow: 0 0 16px currentColor;
}
.chat-content { flex: 1; min-width: 0; }
.chat-name {
    font-size: 11px; font-weight: 600; margin-bottom: 5px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: .05em;
}
.chat-text {
    background: #111120;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 0 16px 16px 16px;
    padding: 12px 16px;
    font-size: 13px;
    line-height: 1.6;
    color: #cbd5e1;
}
.chat-reel {
    margin-top: 8px;
    background: rgba(124,58,237,0.07);
    border: 1px solid rgba(124,58,237,0.18);
    border-radius: 12px;
    padding: 10px 14px;
}
.chat-reel-title {
    font-size: 10px; font-weight: 600; color: #7c3aed;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase; letter-spacing: .08em;
    margin-bottom: 6px;
}
.chat-reaction {
    margin-top: 6px;
    font-size: 12px;
    color: #64748b;
    padding-left: 4px;
    border-left: 2px solid rgba(255,255,255,0.08);
    padding: 4px 0 4px 10px;
}
.reel-bullet { font-size: 13px; color: #94a3b8; margin: 3px 0; }

/* ── History panel ── */
.hist-entry {
    margin-bottom: 7px; padding: 8px 12px;
    border-radius: 10px; background: rgba(124,58,237,0.06);
    border-left: 2px solid rgba(124,58,237,0.25);
}
.hist-q { color: #475569; font-size: 11px; margin-bottom: 2px; }
.hist-a { color: #c084fc; font-weight: 500; font-size: 13px; }

/* ── Persona indicator ── */
.persona-bar {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px;
    background: rgba(124,58,237,0.07);
    border: 1px solid rgba(124,58,237,0.15);
    border-radius: 12px;
    margin-bottom: 12px;
    font-size: 13px;
}

/* ── Intervene panel ── */
.intervene-wrap {
    border: 1px solid rgba(245,158,11,0.2) !important;
    background: rgba(245,158,11,0.04) !important;
    border-radius: 16px !important;
    padding: 16px !important;
}

/* ── Telemetry bar ── */
.telemetry {
    background: #0a0a18;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 8px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #475569;
    text-align: center;
    margin-top: 10px;
}

/* ── Glow score pulse ── */
.score-pulse {
    animation: score-glow 2s ease-in-out infinite alternate;
}
@keyframes score-glow {
    from { opacity: .7; }
    to   { opacity: 1;  }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.3); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,0.6); }
"""

# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

_BG  = "#07070f"
_SRF = "#111120"
_GRD = "rgba(255,255,255,0.04)"
_FNT = dict(color="#94a3b8", family="DM Sans")


def _fig_base(**kw) -> dict:
    return dict(
        paper_bgcolor=_BG, plot_bgcolor=_SRF, font=_FNT,
        margin=dict(l=48, r=48, t=28, b=40),
        **kw,
    )


def reward_chart(result: EpisodeResult) -> go.Figure:
    if not result.steps:
        return go.Figure()
    xs  = [s.step_num for s in result.steps]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="step reward", x=xs,
        y=[s.reward for s in result.steps],
        marker_color=["#22c55e" if s.reward >= 0 else "#ef4444" for s in result.steps],
        marker_line_width=0, opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        name="completeness", x=xs,
        y=[s.completeness for s in result.steps],
        mode="lines+markers",
        line=dict(color="#c084fc", width=2.5, shape="spline"),
        marker=dict(size=8, color="#c084fc",
                    line=dict(color="#07070f", width=2)),
        yaxis="y2",
    ))
    fig.update_layout(
        **_fig_base(height=260),
        legend=dict(bgcolor=_SRF, bordercolor="rgba(255,255,255,0.06)",
                    orientation="h", y=1.12),
        yaxis =dict(title="reward",       gridcolor=_GRD, zerolinecolor=_GRD,
                    titlefont=dict(size=11)),
        yaxis2=dict(title="completeness", overlaying="y", side="right",
                    range=[0,1], gridcolor=_GRD, titlefont=dict(size=11)),
        xaxis =dict(title="step",         gridcolor=_GRD, tickfont=dict(size=11)),
        bargap=0.35,
    )
    return fig


def score_gauge(report: GradeReport) -> go.Figure:
    s = report.final_score
    c = "#22c55e" if s>=.90 else "#7c3aed" if s>=.80 else "#f59e0b" if s>=.70 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(s * 100, 1),
        delta=dict(reference=90, suffix="%",
                   increasing=dict(color="#22c55e"),
                   decreasing=dict(color="#ef4444")),
        number=dict(suffix="%", font=dict(color=c, size=42,
                                          family="Syne")),
        gauge=dict(
            axis=dict(range=[0,100], tickcolor="#334155",
                      tickfont=dict(color="#475569", size=10)),
            bar=dict(color=c, thickness=0.28),
            bgcolor=_SRF, bordercolor="rgba(255,255,255,0.04)",
            steps=[
                dict(range=[0,  60], color="#0d0d1e"),
                dict(range=[60, 80], color="#0f0d20"),
                dict(range=[80, 90], color="#100e24"),
                dict(range=[90,100], color="#0d1a16"),
            ],
            threshold=dict(
                line=dict(color="#c084fc", width=2.5),
                thickness=0.85, value=90,
            ),
        ),
        title=dict(
            text=f"Grade: <b>{report.grade}</b>  ·  Hard pass: {'✓' if report.passed_hard_mode else '✗'}",
            font=dict(color=c, size=13, family="JetBrains Mono"),
        ),
    ))
    fig.update_layout(**_fig_base(height=240,
                                   margin=dict(l=16,r=16,t=16,b=16)))
    return fig


def breakdown_table(report: GradeReport) -> str:
    b = report.breakdown
    return (
        "| component | contribution |\n|---|---|\n"
        f"| completeness `×0.50` | `{b['completeness_contribution']:+.4f}` |\n"
        f"| diversity    `×0.30` | `{b['diversity_contribution']:+.4f}` |\n"
        f"| efficiency   `×0.20` | `{b['efficiency_contribution']:+.4f}` |\n"
        f"| penalty      `−`     | `{-b['penalty_deduction']:+.4f}` |\n"
        f"| **total**             | **`{report.final_score:.4f}`** |"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Classroom chat renderer
# ─────────────────────────────────────────────────────────────────────────────

PERSONA_COLORS = {
    "professor_z": "#7c3aed",
    "sarah":       "#0ea5e9",
    "aman":        "#f59e0b",
    "bud_ai":      "#22c55e",
}

PERSONA_AVATARS = {
    "professor_z": "👨‍🏫",
    "sarah":       "⚡",
    "aman":        "🤓",
    "bud_ai":      "🦾",
}


def _render_message(msg: ClassroomMessage) -> str:
    pk    = msg.persona.value
    color = PERSONA_COLORS.get(pk, "#7c3aed")
    ava   = PERSONA_AVATARS.get(pk, "🤖")

    # Study reel
    reel_html = ""
    if msg.study_reel:
        bullets = "".join(
            f'<div class="reel-bullet">⚡ {b}</div>'
            for b in msg.study_reel
        )
        reel_html = (
            f'<div class="chat-reel">'
            f'<div class="chat-reel-title">study reel</div>'
            f'{bullets}'
            f'</div>'
        )

    # Social reaction
    react_html = ""
    if msg.social_reaction:
        src = msg.social_reaction.get("from", "")
        txt = msg.social_reaction.get("message", "")
        if src and txt:
            react_html = (
                f'<div class="chat-reaction">'
                f'<span style="color:#64748b;font-size:11px">{src}</span> '
                f'<span style="color:#94a3b8">{txt}</span>'
                f'</div>'
            )

    # Vibe hint
    vibe  = msg.ui_hint.get("vibe_label", "")
    glow  = msg.ui_hint.get("glow_color", color)
    vibe_html = ""
    if vibe:
        vibe_html = (
            f'<div style="margin-top:7px;display:inline-block;'
            f'font-size:10px;font-family:\'JetBrains Mono\',monospace;'
            f'color:{glow};background:rgba(0,0,0,0.3);'
            f'border:1px solid {glow}40;border-radius:20px;padding:2px 10px;">'
            f'◉ {vibe}</div>'
        )

    return (
        f'<div class="chat-bubble">'
        f'  <div class="chat-avatar" style="background:{color}18;color:{color};">{ava}</div>'
        f'  <div class="chat-content">'
        f'    <div class="chat-name" style="color:{color}">{msg.persona_display}</div>'
        f'    <div class="chat-text">{msg.dialogue}</div>'
        f'    {reel_html}{react_html}{vibe_html}'
        f'  </div>'
        f'</div>'
    )


def render_classroom_chat(messages: list[ClassroomMessage]) -> str:
    if not messages:
        return (
            '<div style="text-align:center;padding:40px 20px;color:#475569;'
            'font-family:\'DM Sans\',sans-serif;">'
            '<div style="font-size:28px;margin-bottom:10px">🎓</div>'
            '<div style="font-size:14px">run a simulation to start the classroom</div>'
            '</div>'
        )
    bubbles = "".join(_render_message(m) for m in messages)
    return f'<div class="chat-wrap">{bubbles}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def do_run_simulation(
    agent_key: str,
    mode_key:  str,
    cx:        str,
    mem:       DualLayerMemory,
) -> tuple:
    metrics = reset_session()
    classroom_msgs: list[ClassroomMessage] = []

    try:
        mode = Mode(mode_key)

        # Build agent
        use_classroom = agent_key == "classroom" and cfg.has_api_key
        use_gemini    = agent_key == "gemini"    and cfg.has_api_key

        if use_classroom:
            agent = CognitiveClassroom(complexity_mode=ComplexityMode(cx), memory=mem)
        elif use_gemini:
            agent = MemoryAgent(complexity_mode=ComplexityMode(cx), memory=mem)
        else:
            agent = HeuristicAgent()

        # Custom run loop to capture classroom messages
        from core.environment.engine import StudyEnv
        from core.grading.grader import grade
        from core.benchmark import EpisodeResult, EpisodeStep

        env    = StudyEnv(mode=mode)
        state  = env.reset()
        result = EpisodeResult(mode=mode_key, agent_name=agent.name)
        agent.reset()
        step_num = 0

        while not state.is_terminal:
            step_num += 1
            response  = agent.act(state)
            sr        = env.step(response.action)
            result.total_reward += sr.reward
            result.steps.append(EpisodeStep(
                step_num=step_num, action=response.action.value,
                reasoning=response.reasoning, reward=sr.reward,
                completeness=sr.state.completeness,
                diversity=sr.state.diversity_score,
                steps_left=sr.state.steps_left,
                latency_ms=response.latency_ms,
                penalties=sr.info.get("penalties", []),
                bonuses=sr.info.get("bonuses", []),
            ))
            # Capture classroom messages if classroom agent
            if use_classroom and hasattr(agent, "last_message") and agent.last_message:
                classroom_msgs.append(agent.last_message)
            # Update episodic memory
            if hasattr(agent, "record_outcome"):
                agent.record_outcome(response.action.value, sr.reward)
            elif hasattr(agent, "memory"):
                agent.memory.record_episode(response.action.value, sr.reward)

            metrics.record_call(tokens=0, latency_ms=response.latency_ms)
            state = sr.state

        result.final_state = state
        result.report      = grade(state, result.total_reward)

        facts = mem.all_facts() if hasattr(agent, "memory") else []
        metrics.facts_extracted = len(facts)

        report    = result.report
        log_lines = result.to_log_lines()
        notes_txt = result.final_state.notes if result.final_state else "—"

        score_md = (
            f"**{report.final_score:.4f}** · grade **{report.grade}** · "
            f"hard pass {'✓' if report.passed_hard_mode else '✗'}"
            if report else "simulation failed"
        )

        facts_txt = "\n".join(
            f"{'[H]' if f.source=='human' else '[A]'} {f.concept}: {f.definition[:80]}"
            for f in facts
        ) or "no facts extracted yet"

        chat_html = render_classroom_chat(classroom_msgs) if classroom_msgs else (
            "<div style='color:#475569;font-size:13px;padding:20px;text-align:center'>"
            "Classroom mode requires the Cognitive Classroom agent with a Gemini API key.</div>"
        )

        return (
            "\n".join(log_lines),
            score_md,
            notes_txt,
            reward_chart(result),
            score_gauge(report) if report else go.Figure(),
            breakdown_table(report) if report else "",
            build_graph(facts),
            facts_txt,
            chat_html,
            f'<div class="telemetry">{metrics.ui_footer()}</div>',
        )

    except Exception as exc:
        log.exception("simulation error")
        empty = go.Figure()
        err   = f"ERROR: {exc}\n\n{traceback.format_exc()}"
        return (err, "error", "", empty, empty, "", empty, "",
                f"<pre style='color:#ef4444;font-size:11px'>{exc}</pre>",
                '<div class="telemetry">error</div>')


def do_run_comparison(agent_key: str, mem: DualLayerMemory) -> tuple[pd.DataFrame, str]:
    agents = [HeuristicAgent()]
    if cfg.has_api_key:
        try:
            agents.append(MemoryAgent(memory=mem))
        except Exception:
            pass
    rows = run_comparative_benchmark(agents, modes=list(Mode), runs_per_combo=1)
    df   = pd.DataFrame(rows) if rows else pd.DataFrame([{"info": "no results — check API key"}])
    return df, f"completed {len(rows)} benchmark combinations  ·  hard-mode pass threshold: 0.90 (S)"


# ─────────────────────────────────────────────────────────────────────────────
# Conversation tree rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_left(state: ConversationState) -> tuple[str, str, list, bool]:
    hist_html = "".join(
        f'<div class="hist-entry">'
        f'<div class="hist-q">{q}</div>'
        f'<div class="hist-a">→ {a}</div>'
        f'</div>'
        for q, a in state.history
    )
    if TREE.is_terminal(state):
        return hist_html, "✅ launching simulation…", [], False
    node = TREE.current(state)
    opts = [f"{o.icon} {o.label}".strip() for o in node.options]
    if node.type == NodeType.QUESTION and node.other_label:
        opts.append(f"✏️  {node.other_label}")
    return hist_html, node.prompt, opts, node.type == NodeType.FREE_TEXT


# ─────────────────────────────────────────────────────────────────────────────
# App builder
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:

    with gr.Blocks(
        title  = "OpenEnv · Study Intelligence",
        css    = CSS,
        theme  = gr.themes.Base(
            primary_hue  = gr.themes.colors.violet,
            neutral_hue  = gr.themes.colors.slate,
            font         = [gr.themes.GoogleFont("DM Sans"), "system-ui"],
            font_mono    = [gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
        ),
    ) as demo:

        # ── Persistent state ──────────────────────────────────────────────
        conv_st  = gr.State(TREE.start())
        mem_st   = gr.State(DualLayerMemory(episodic_window=5))
        agent_st = gr.State("classroom" if cfg.has_api_key else "heuristic")
        mode_st  = gr.State("hard")
        cx_st    = gr.State("standard")

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML(f"""
        <div style="padding:22px 4px 16px;border-bottom:1px solid rgba(255,255,255,0.05);
             margin-bottom:20px;display:flex;align-items:center;justify-content:space-between;">
          <div>
            <div style="display:flex;align-items:baseline;gap:14px;">
              <span style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;
                    background:linear-gradient(135deg,#c084fc,#7c3aed,#0ea5e9);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;">
                OpenEnv
              </span>
              <span style="font-family:'JetBrains Mono',monospace;font-size:11px;
                    color:#475569;letter-spacing:.04em;">
                v{cfg.spec_version} · cognitive classroom
              </span>
            </div>
            <div style="font-size:13px;color:#64748b;margin-top:5px;font-weight:300;">
              autonomous study intelligence benchmark
            </div>
          </div>
          <div style="text-align:right;font-family:'JetBrains Mono',monospace;font-size:11px;
               line-height:1.7;color:#475569;">
            <div style="color:{'#22c55e' if cfg.has_api_key else '#f59e0b'}">
              {'● gemini connected' if cfg.has_api_key else '● heuristic mode (no api key)'}
            </div>
            <div>S = 0.50C + 0.30D + 0.20E − P</div>
          </div>
        </div>
        """)

        with gr.Row(equal_height=False, variant="panel"):

            # ── LEFT: guided setup ─────────────────────────────────────────
            with gr.Column(scale=1, min_width=290):

                gr.HTML("""
                <p style="color:#475569;font-size:10px;font-family:'JetBrains Mono',monospace;
                   letter-spacing:.08em;text-transform:uppercase;margin-bottom:12px;">
                  ▸ guided setup
                </p>""")

                hist_out     = gr.HTML("")
                question_out = gr.Markdown("👋 what do you want to do?",
                                           elem_classes=["markdown-body"])

                opt_btns = [gr.Button(visible=False, elem_classes=["btn-choice"])
                            for _ in range(7)]

                with gr.Group(visible=False) as free_grp:
                    free_in  = gr.Textbox(
                        placeholder="describe your goal here…",
                        show_label=False, lines=2,
                    )
                    free_btn = gr.Button("submit →", elem_classes=["btn-primary"])

                gr.HTML('<div style="margin:12px 0;border-top:1px solid rgba(255,255,255,0.05)"></div>')
                reset_btn = gr.Button("↺ start over", elem_classes=["btn-secondary"])

                # Complexity mode
                gr.HTML("""
                <p style="color:#475569;font-size:10px;font-family:'JetBrains Mono',monospace;
                   letter-spacing:.08em;text-transform:uppercase;margin:16px 0 8px;">
                  ▸ complexity mode
                </p>""")
                cx_radio = gr.Radio(
                    choices=[("ELI5 🧒", "eli5"),
                             ("Standard 🎓", "standard"),
                             ("PhD 🔬", "phd")],
                    value="standard", show_label=False,
                )
                gr.HTML("""
                <p style="color:#334155;font-size:11px;margin:5px 0 0;line-height:1.5;">
                  one environment · infinite learning levels
                </p>""")

                # Agent selector
                gr.HTML("""
                <p style="color:#475569;font-size:10px;font-family:'JetBrains Mono',monospace;
                   letter-spacing:.08em;text-transform:uppercase;margin:16px 0 8px;">
                  ▸ agent
                </p>""")
                agent_radio = gr.Radio(
                    choices=[
                        ("🎓 Cognitive Classroom", "classroom"),
                        ("🧠 Memory Agent",        "gemini"),
                        ("🧮 Heuristic Baseline",  "heuristic"),
                    ],
                    value="classroom" if cfg.has_api_key else "heuristic",
                    show_label=False,
                )

            # ── RIGHT: output ──────────────────────────────────────────────
            with gr.Column(scale=2):

                with gr.Tabs():

                    # ── Classroom chat ─────────────────────────────────────
                    with gr.Tab("🎓 classroom"):
                        gr.HTML("""
                        <p style="color:#475569;font-size:12px;margin-bottom:12px;line-height:1.5;">
                          Professor Z · Sarah · Aman · Bud AI take turns based on your learning score.
                          Each brings a different style. They bicker. It keeps things real.
                        </p>""")
                        chat_out = gr.HTML(
                            '<div style="text-align:center;padding:50px 20px;color:#334155;'
                            'font-size:14px;">run a simulation to start the classroom 🎓</div>'
                        )

                    # ── Simulation analytics ───────────────────────────────
                    with gr.Tab("⚡ analytics"):
                        score_out = gr.Markdown(
                            "run a simulation to see results",
                            elem_classes=["markdown-body"],
                        )
                        with gr.Row():
                            gauge_out  = gr.Plot(show_label=False)
                            reward_out = gr.Plot(show_label=False)
                        bdown_out  = gr.Markdown("", elem_classes=["markdown-body"])
                        log_out    = gr.Textbox(
                            label="agent decision log",
                            lines=10, interactive=False,
                            show_copy_button=True,
                        )
                        notes_out = gr.Textbox(
                            label="final knowledge base",
                            lines=6, interactive=False,
                            show_copy_button=True,
                        )

                    # ── Knowledge graph ────────────────────────────────────
                    with gr.Tab("🧠 knowledge graph"):
                        gr.HTML("""
                        <p style="color:#475569;font-size:12px;margin-bottom:10px;">
                          every [FACT] the agent extracts becomes a node.
                          gold nodes = human corrections. the graph grows in real time.
                        </p>""")
                        graph_out = gr.Plot(show_label=False)
                        facts_out = gr.Textbox(
                            label="learned concepts", lines=8, interactive=False,
                        )

                    # ── Human-in-the-loop ──────────────────────────────────
                    with gr.Tab("🤝 intervene"):
                        gr.Markdown(
                            "### Human-in-the-Loop Correction\n\n"
                            "Your correction is stored in semantic memory at confidence **1.0** "
                            "and cannot be overwritten by the agent. "
                            "The classroom picks it up on the next run.\n\n"
                            "Use `[FACT: concept | definition]` tags for structured extraction.",
                            elem_classes=["markdown-body"],
                        )
                        with gr.Group(elem_classes=["intervene-wrap"]):
                            iv_concept = gr.Textbox(
                                label="concept / topic",
                                placeholder="e.g.  backpropagation",
                            )
                            iv_text = gr.Textbox(
                                label="your correction or fact",
                                placeholder="[FACT: backpropagation | Algorithm for computing gradients in neural nets]",
                                lines=4,
                            )
                            iv_btn = gr.Button(
                                "💾 store in semantic memory",
                                elem_classes=["btn-primary"],
                            )
                        iv_status = gr.Markdown("", elem_classes=["markdown-body"])

                    # ── Comparison ─────────────────────────────────────────
                    with gr.Tab("⚖️ comparison"):
                        gr.Markdown(
                            "Runs all **agents × modes**. Use this table during your pitch: "
                            "*'our environment proves model X outperforms Y under constraint Z.'*",
                            elem_classes=["markdown-body"],
                        )
                        cmp_btn   = gr.Button(
                            "run full benchmark matrix →",
                            elem_classes=["btn-primary"],
                        )
                        cmp_note  = gr.Markdown("")
                        cmp_table = gr.DataFrame(
                            headers=["Model","Mode","Avg Score","Best Score",
                                     "Worst Score","Runs","Hard Pass"],
                            label="results",
                        )

                    # ── About ──────────────────────────────────────────────
                    with gr.Tab("📖 about"):
                        gr.Markdown(f"""
### OpenEnv Study Intelligence v{cfg.spec_version}

**Problem:** LLMs score well on MMLU. They fail on stateful, long-horizon tasks
where past decisions compound and constraints tighten — exactly how real learning works.

**Solution:** Separate *strategy* (OpenEnv) from *execution* (LLM).
The environment decides **what to do**. The LLM decides **how to write it**.

#### The Cognitive Classroom

| Persona | Triggers when | Personality |
|---|---|---|
| 👨‍🏫 Professor Z | completeness < 0.40 | Calm strategist — builds foundations |
| ⚡ Sarah | 0.40 ≤ score ≤ 0.70 | Flow queen — creates study reels |
| 🤓 Aman | completeness > 0.70 | Technical rival — quizzes hard |
| 🦾 Bud AI | always | Loyal assistant — tracks score |

#### Score formula
```
S = (0.50 × C) + (0.30 × D) + (0.20 × E) − P
```

#### Reference scores (hard mode)
| model | score | grade |
|---|---|---|
| GPT-4o | 0.92 | S |
| Gemini 1.5 Pro | 0.88 | A |
| Llama 3 8B | 0.65 | C |
| Heuristic | 0.78 | B |

**Hard-mode pass threshold: ≥ 0.90**
                        """, elem_classes=["markdown-body"])

                # Telemetry footer
                telem_out = gr.HTML(
                    '<div class="telemetry">telemetry appears after first run</div>'
                )

        # ─── Output slot definitions ───────────────────────────────────────
        LEFT_OUTS = [conv_st, hist_out, question_out, free_grp] + opt_btns

        RUN_OUTS = [
            log_out, score_out, notes_out,
            reward_out, gauge_out, bdown_out,
            graph_out, facts_out, chat_out, telem_out,
        ]
        EMPTY_RUN = [""] * len(RUN_OUTS)

        STATE_OUTS = [agent_st, mode_st, cx_st]

        # ─── Helpers ──────────────────────────────────────────────────────
        def _left_updates(new_st: ConversationState) -> list:
            h, q, opts, show_free = render_left(new_st)
            btn_ups = [
                gr.update(value=opts[i], visible=True) if i < len(opts)
                else gr.update(visible=False)
                for i in range(len(opt_btns))
            ]
            return [new_st, h, q, gr.update(visible=show_free)] + btn_ups

        def _maybe_autorun(new_st, mem, ak, mk, cx) -> list:
            if not TREE.is_terminal(new_st):
                return EMPTY_RUN
            try:
                node = TREE.get_node(new_st.current_node_id)
                if node.action_key == "run_simulation":
                    return list(do_run_simulation(ak, mk, cx, mem))
            except Exception:
                pass
            return EMPTY_RUN

        # ─── Handlers ─────────────────────────────────────────────────────
        def on_option(label, st, mem, ak, mk, cx):
            node = TREE.current(st)
            if node.other_label and label.endswith(node.other_label.strip()):
                new_st = TREE.advance(st, "other")
            else:
                matched = next(
                    (o for o in node.options if label.strip().endswith(o.label.strip())),
                    None,
                )
                if not matched:
                    return _left_updates(st) + EMPTY_RUN + [ak, mk, cx]
                if node.id == "pick_agent": ak = matched.key
                if node.id == "pick_mode":  mk = matched.key
                new_st = TREE.advance(st, matched.key)
            return _left_updates(new_st) + _maybe_autorun(new_st, mem, ak, mk, cx) + [ak, mk, cx]

        def on_free(text, st, mem, ak, mk, cx):
            new_st = TREE.advance(st, "other", free_text=text)
            return _left_updates(new_st) + _maybe_autorun(new_st, mem, ak, mk, cx) + [ak, mk, cx]

        def on_reset(st, mem):
            new_mem = DualLayerMemory(episodic_window=5)
            new_st  = TREE.start()
            return (_left_updates(new_st) + EMPTY_RUN + [new_mem]
                    + ["classroom" if cfg.has_api_key else "heuristic", "hard", "standard"])

        def on_cx(val, _):   return val
        def on_agent(val, _): return val

        def on_intervene(concept, text, mem):
            if not text.strip():
                return mem, "⚠ nothing to store — type your correction first"
            facts = mem.extract_and_store_facts(text, source="human")
            if not facts and concept.strip():
                mem.store_fact(concept.strip(), text[:200], source="human", confidence=1.0)
                facts = [f for f in [mem.retrieve_fact(concept.strip())] if f]
            if facts:
                names = ", ".join(f"`{f.concept}`" for f in facts)
                return (mem,
                    f"✓ stored **{len(facts)}** fact(s): {names}\n\n"
                    f"Injected into the classroom on next run.")
            return mem, "⚠ no facts found — use `[FACT: concept | definition]` tags"

        def on_compare(mem, ak):
            df, note = do_run_comparison(ak, mem)
            return df, note

        # ─── Wire everything ───────────────────────────────────────────────
        BTN_INPUTS = [conv_st, mem_st, agent_st, mode_st, cx_st]

        for btn in opt_btns:
            btn.click(
                fn=on_option,
                inputs=[btn] + BTN_INPUTS,
                outputs=LEFT_OUTS + RUN_OUTS + STATE_OUTS,
            )

        free_btn.click(
            fn=on_free,
            inputs=[free_in] + BTN_INPUTS,
            outputs=LEFT_OUTS + RUN_OUTS + STATE_OUTS,
        )

        reset_btn.click(
            fn=on_reset,
            inputs=[conv_st, mem_st],
            outputs=LEFT_OUTS + RUN_OUTS + [mem_st] + STATE_OUTS,
        )

        cx_radio.change(fn=on_cx,    inputs=[cx_radio,    cx_st],    outputs=[cx_st])
        agent_radio.change(fn=on_agent, inputs=[agent_radio, agent_st], outputs=[agent_st])

        iv_btn.click(
            fn=on_intervene,
            inputs=[iv_concept, iv_text, mem_st],
            outputs=[mem_st, iv_status],
        )

        cmp_btn.click(
            fn=on_compare,
            inputs=[mem_st, agent_st],
            outputs=[cmp_table, cmp_note],
        )

        demo.load(
            fn=lambda s: render_left(s)[1:],
            inputs=[conv_st],
            outputs=[question_out, free_grp] + opt_btns,
        )

    return demo


if __name__ == "__main__":
    log.info(f"OpenEnv v{cfg.spec_version} — Liquid Cognitive Classroom")
    log.info(f"api key: {'present' if cfg.has_api_key else 'MISSING — heuristic fallback'}")
    build_app().launch(
        server_name = cfg.server.host,
        server_port = cfg.server.port,
        show_error  = True,
    )
