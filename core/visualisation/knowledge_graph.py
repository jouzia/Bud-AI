"""
Knowledge Graph Visualisation

Renders the agent's growing semantic memory as an interactive Plotly network.

Each [FACT] extracted by the agent becomes a node.
Edges represent co-occurrence within the same step (heuristic proximity).
Human-corrected facts are highlighted in a distinct colour.

Design: pure function build_graph() — no state, no side effects.
        Pass it the list of facts and get a Figure back.
"""
from __future__ import annotations

import math
from typing import Optional

import plotly.graph_objects as go

from core.memory.memory import Fact

# ── Colour scheme ─────────────────────────────────────────────────────────────
_COLOURS = {
    "agent":  "#7c3aed",   # purple — agent-extracted
    "human":  "#f59e0b",   # gold   — human-corrected (high value)
    "edge":   "#2a2a4a",
    "bg":     "#09090f",
    "surface":"#141420",
    "text":   "#e2e8f0",
    "subtext":"#94a3b8",
}


def _circular_layout(n: int, radius: float = 1.0) -> list[tuple[float, float]]:
    """Evenly distribute n nodes on a circle."""
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0)]
    return [
        (radius * math.cos(2 * math.pi * i / n),
         radius * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]


def build_graph(
    facts:      list[Fact],
    title:      str = "knowledge graph",
    width:      int = 600,
    height:     int = 420,
) -> go.Figure:
    """
    Build a Plotly network graph from a list of Fact objects.
    Returns a Figure ready to drop into gr.Plot().
    """
    if not facts:
        return _empty_figure(title, width, height)

    positions = _circular_layout(len(facts), radius=2.0)

    # ── Nodes ────────────────────────────────────────────────────────────────
    node_x, node_y, node_text, node_hover, node_colors, node_sizes = [], [], [], [], [], []

    for i, fact in enumerate(facts):
        x, y = positions[i]
        node_x.append(x)
        node_y.append(y)
        # Truncate label for the graph — full definition in hover
        label = fact.concept if len(fact.concept) <= 16 else fact.concept[:14] + "…"
        node_text.append(label)
        node_hover.append(
            f"<b>{fact.concept}</b><br>"
            f"{fact.definition[:120]}{'…' if len(fact.definition) > 120 else ''}<br>"
            f"<i>source: {fact.source} | confidence: {fact.confidence:.0%}</i>"
        )
        node_colors.append(_COLOURS["human"] if fact.source == "human" else _COLOURS["agent"])
        node_sizes.append(22 if fact.source == "human" else 18)

    # ── Edges (connect adjacent concepts — simple proximity heuristic) ───────
    edge_x, edge_y = [], []
    for i in range(len(facts) - 1):
        x0, y0 = positions[i]
        x1, y1 = positions[i + 1]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure()

    # Draw edges first (below nodes)
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color=_COLOURS["edge"], width=1),
        hoverinfo="none",
        showlegend=False,
    ))

    # Draw nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(color="#ffffff", width=1.5),
            opacity=0.92,
        ),
        text=node_text,
        textposition="top center",
        textfont=dict(color=_COLOURS["text"], size=10, family="DM Sans"),
        hovertext=node_hover,
        hoverinfo="text",
        showlegend=False,
    ))

    # Legend markers
    for source, colour, label in [
        ("agent", _COLOURS["agent"], "agent fact"),
        ("human", _COLOURS["human"], "human correction"),
    ]:
        if any(f.source == source for f in facts):
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(color=colour, size=10),
                name=label,
                showlegend=True,
            ))

    fig.update_layout(
        title=dict(
            text=f"{title}  ({len(facts)} concepts)",
            font=dict(color=_COLOURS["subtext"], size=13, family="DM Sans"),
            x=0.01,
        ),
        paper_bgcolor = _COLOURS["bg"],
        plot_bgcolor  = _COLOURS["surface"],
        showlegend    = True,
        legend=dict(
            bgcolor     = _COLOURS["surface"],
            bordercolor = _COLOURS["edge"],
            borderwidth = 1,
            font        = dict(color=_COLOURS["text"], size=11),
            orientation = "h",
            y=-0.12,
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=60),
        width=width,
        height=height,
        hovermode="closest",
    )

    return fig


def _empty_figure(title: str, width: int, height: int) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="No concepts learned yet.<br>Run a simulation to populate the graph.",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(color="#94a3b8", size=13, family="DM Sans"),
    )
    fig.update_layout(
        paper_bgcolor="#09090f",
        plot_bgcolor ="#141420",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        width=width, height=height,
        title=dict(text=title, font=dict(color="#94a3b8", size=13)),
    )
    return fig
