"""
Benchmark Runner

Separates the orchestration logic from the Gradio UI entirely.
The UI calls run_episode() and gets back structured results.
No Gradio imports here — pure Python, fully testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.agents.base import BaseAgent, AgentResponse
from core.environment.engine import StudyEnv, StepResult
from core.environment.state import Mode, StudyState
from core.grading.grader import GradeReport, grade


@dataclass
class EpisodeStep:
    step_num:   int
    action:     str
    reasoning:  str
    reward:     float
    completeness: float
    diversity:  float
    steps_left: int
    latency_ms: float
    penalties:  list[str]
    bonuses:    list[str]


@dataclass
class EpisodeResult:
    mode:         str
    agent_name:   str
    steps:        list[EpisodeStep] = field(default_factory=list)
    final_state:  StudyState | None = None
    total_reward: float = 0.0
    report:       GradeReport | None = None
    error:        str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    def to_log_lines(self) -> list[str]:
        lines = []
        for s in self.steps:
            status = "✓" if s.reward >= 0 else "✗"
            lines.append(
                f"  {status} Step {s.step_num:02d} │ "
                f"[{s.action.upper():<12}] │ "
                f"R={s.reward:+.3f} │ "
                f"C={s.completeness:.2f} │ "
                f"Steps left={s.steps_left} │ "
                f"{s.latency_ms:.0f}ms"
            )
            if s.reasoning:
                lines.append(f"       💭 {s.reasoning}")
        return lines


def run_episode(
    agent:   BaseAgent,
    mode:    Mode | str = Mode.HARD,
    seed:    int | None = None,
    verbose: bool = False,
) -> EpisodeResult:
    """
    Run a single episode. Returns structured EpisodeResult.
    The UI layer formats this however it wants.
    """
    mode   = Mode(mode) if isinstance(mode, str) else mode
    env    = StudyEnv(mode=mode, seed=seed)
    state  = env.reset()
    result = EpisodeResult(mode=mode.value, agent_name=agent.name)
    agent.reset()

    try:
        step_num = 0
        while not state.is_terminal:
            step_num += 1
            response: AgentResponse = agent.act(state)
            sr: StepResult          = env.step(response.action)

            result.total_reward += sr.reward
            result.steps.append(EpisodeStep(
                step_num     = step_num,
                action       = response.action.value,
                reasoning    = response.reasoning,
                reward       = sr.reward,
                completeness = sr.state.completeness,
                diversity    = sr.state.diversity_score,
                steps_left   = sr.state.steps_left,
                latency_ms   = response.latency_ms,
                penalties    = sr.info.get("penalties", []),
                bonuses      = sr.info.get("bonuses", []),
            ))

            state = sr.state

            if verbose:
                print(result.to_log_lines()[-1])

        result.final_state = state
        result.report      = grade(state, result.total_reward)

    except Exception as exc:
        result.error = str(exc)

    return result


def run_comparative_benchmark(
    agents: list[BaseAgent],
    modes:  list[Mode] | None = None,
    runs_per_combo: int = 1,
) -> list[dict]:
    """
    Run all agent × mode combinations.
    Returns list of dicts suitable for pandas DataFrame / display table.
    """
    modes  = modes or list(Mode)
    rows   = []

    for agent in agents:
        for mode in modes:
            scores = []
            for seed in range(runs_per_combo):
                r = run_episode(agent, mode=mode, seed=seed)
                if r.succeeded and r.report:
                    scores.append(r.report.final_score)

            if scores:
                import numpy as np
                rows.append({
                    "Model":        agent.name,
                    "Mode":         mode.value,
                    "Avg Score":    round(float(np.mean(scores)), 4),
                    "Best Score":   round(float(np.max(scores)), 4),
                    "Worst Score":  round(float(np.min(scores)), 4),
                    "Runs":         len(scores),
                    "Hard Pass":    any(s >= 0.90 for s in scores) if mode == Mode.HARD else "N/A",
                })

    return rows
