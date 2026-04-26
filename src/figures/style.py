"""
Shared plot styling for Nature Sustainability / Nature format.

Nature guidelines:
- Single column: 89mm (3.5in), double column: 183mm (7.2in)
- Font: sans-serif, minimum 5pt in final print
- Resolution: 300 dpi minimum
- File format: PDF or EPS preferred, PNG at 300+ dpi acceptable
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Nature figure dimensions ─────────────────────────────────
SINGLE_COL = 3.5    # inches
DOUBLE_COL = 7.2    # inches
FULL_PAGE_H = 9.0   # inches (max height)

# ── Species colors (consistent across all figures) ───────────
COLORS = {
    "compliance":  "#2166AC",
    "intermediary": "#E31A1C",
    "bct":         "#FF7F00",
    "habitatbank": "#6A3D9A",
}

# ── Status colors (early warning dashboard) ──────────────────
STATUS = {
    "safe":     "#4DAF4A",
    "warning":  "#FF7F00",
    "critical": "#E31A1C",
}


def set_nature_style():
    """Apply Nature-compatible matplotlib style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 1.0,
        "axes.grid": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })


PCT_FMT = plt.FuncFormatter(lambda v, _: f"{v:.0%}")


def panel_label(ax, label, x=-0.12, y=1.08):
    """Add panel label (a, b, c...) in Nature style."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")
