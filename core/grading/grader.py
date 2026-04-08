"""
OpenEnv Grader

Final score formula:
    S = (w_c * C) + (w_d * D) + (w_e * E) - P

Where:
    C = completeness (0–1)
    D = diversity score (0–1)
    E = efficiency = steps_used / step_limit (lower steps for same result = higher score)
    P = cumulative penalty
    w_* = configurable weights
"""
from __future__ import annotations

from dataclasses import dataclass

from core.environment.state import StudyState, MODE_STEP_LIMITS


@dataclass(frozen=True)
class GradeReport:
    final_score:       float
    completeness:      float
    diversity:         float
    efficiency:        float
    penalty:           float
    grade:             str
    breakdown:         dict[str, float]
    passed_hard_mode:  bool

    def __str__(self) -> str:
        lines = [
            f"━━ OpenEnv Grade Report ━━",
            f"  Final Score   : {self.final_score:.4f}",
            f"  Grade         : {self.grade}",
            f"  Completeness  : {self.completeness:.4f}",
            f"  Diversity     : {self.diversity:.4f}",
            f"  Efficiency    : {self.efficiency:.4f}",
            f"  Penalty       : -{self.penalty:.4f}",
            f"  Hard Mode Pass: {'✓' if self.passed_hard_mode else '✗'}",
        ]
        return "\n".join(lines)


# Weights — must sum to 1.0
WEIGHTS = {
    "completeness": 0.50,
    "diversity":    0.30,
    "efficiency":   0.20,
}


def grade(state: StudyState, total_reward: float) -> GradeReport:
    """
    Produces a deterministic grade from the terminal state.
    total_reward is used only for validation; the grade itself is
    computed from the state directly to avoid reward-hacking.
    """
    step_limit = MODE_STEP_LIMITS[state.mode]
    steps_used = state.steps_used

    # Efficiency: reward using fewer steps to reach high completeness
    efficiency = (
        (state.completeness / max(steps_used, 1)) * step_limit
    ) if steps_used > 0 else 0.0
    efficiency = min(1.0, efficiency)

    raw = (
        WEIGHTS["completeness"] * state.completeness
        + WEIGHTS["diversity"]  * state.diversity_score
        + WEIGHTS["efficiency"] * efficiency
        - state.penalty
    )
    final = round(max(0.0, min(1.0, raw)), 4)

    grade_str = _letter_grade(final)
    passed    = final >= 0.90 and state.mode.value == "hard"

    return GradeReport(
        final_score      = final,
        completeness     = state.completeness,
        diversity        = state.diversity_score,
        efficiency       = efficiency,
        penalty          = state.penalty,
        grade            = grade_str,
        breakdown        = {
            "completeness_contribution": round(WEIGHTS["completeness"] * state.completeness, 4),
            "diversity_contribution":    round(WEIGHTS["diversity"]    * state.diversity_score, 4),
            "efficiency_contribution":   round(WEIGHTS["efficiency"]   * efficiency, 4),
            "penalty_deduction":         round(state.penalty, 4),
        },
        passed_hard_mode = passed,
    )


def _letter_grade(score: float) -> str:
    thresholds = [(0.95, "S+"), (0.90, "S"), (0.80, "A"), (0.70, "B"), (0.60, "C"), (0.0, "F")]
    for threshold, letter in thresholds:
        if score >= threshold:
            return letter
    return "F"
