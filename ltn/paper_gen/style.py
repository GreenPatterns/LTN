"""Shared figure style: dataviz-skill reference palette (light mode, print), fixed
entity->color assignment (never cycled), thin marks, recessive axes."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated reference palette (dataviz skill, light mode)
BLUE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA, ORANGE = (
    "#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"

MODEL_COLORS = {
    "ltn": BLUE, "ltn_gaussian": VIOLET, "ltn_ode": RED,
    "lstm": AQUA, "tgn": YELLOW, "transformer": GREEN,
    "arima": ORANGE, "ewma": MAGENTA, "persistence": INK2,
}
POLICY_COLORS = {
    "ltn_lcb": BLUE, "ltn_mean": AQUA, "greedy_claimed": ORANGE,
    "random": INK2, "oracle": INK,
}
MODEL_LABELS = {
    "ltn": "LTN", "ltn_v2": "LTN-v2", "ltn_gaussian": "LTN (Gaussian state)",
    "ltn_ode": "LTN (neural ODE)",
    "ltn_nograph": "LTN ($-$graph)", "ltn_nophys": "LTN ($-$physics-$\\gamma$)",
    "lstm": "LSTM", "tgn": "TGN", "transformer": "Transformer",
    "arima": "ARIMA", "ewma": "EWMA", "persistence": "Persistence",
}
MODEL_COLORS.update({"ltn_nograph": MAGENTA, "ltn_nophys": ORANGE, "ltn_v2": GREEN})


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.labelcolor": INK,
        "legend.frameon": False, "legend.fontsize": 7,
        "lines.linewidth": 1.6,
    })


def fmt3(x: float) -> str:
    """3 significant digits for table cells."""
    if x != x:
        return "--"
    return f"{x:.3g}"
