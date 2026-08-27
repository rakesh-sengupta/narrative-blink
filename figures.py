"""
figures.py
----------
Builds the four figures used in the paper. Run simulations.py first.

Usage:  python3 figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import PARAMS, availability, cost_profiles, lag_curve, p_report, parse_trace

plt.rcParams.update({
    "font.size": 9, "font.family": "serif", "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 200, "savefig.bbox": "tight",
})

C = {"item": "#4C6EF5", "surprisal": "#E8590C", "integration": "#7048E8",
     "base": "#495057"}
LAGS = [1, 2, 3, 4, 5, 6, 8]
PROF, INDEX = cost_profiles(1.0)
R = json.load(open("results.json"))


# ---------------------------------------------------------------------------
def figure1():
    """The commitment model: availability over time, and the lag curve it implies."""
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))

    t = np.linspace(0, 900, 900)
    for z, lab, sty in [(-1.0, "low-cost T1", ":"), (0.0, "average T1", "-"),
                        (2.0, "high-cost T1", "--")]:
        ax[0].plot(t, availability(t, z), sty, color=C["base"], lw=1.4, label=lab)
    ax[0].set_xlabel("time after T1 onset (ms)")
    ax[0].set_ylabel("uptake capacity available")
    ax[0].set_ylim(-0.02, 1.08)
    ax[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    ax[0].set_title("A  Commitment suspends uptake", loc="left", fontsize=9)

    x = np.arange(len(LAGS))
    for z, lab, sty in [(0.0, "average T1", "-o"), (2.0, "high-cost T1", "--s")]:
        ax[1].plot(x, lag_curve(LAGS, z), sty, color=C["base"], lw=1.4, ms=3.5,
                   mfc="white" if sty.startswith("--") else C["base"], label=lab)
    ax[1].set_xticks(x); ax[1].set_xticklabels(LAGS)
    ax[1].set_xlabel("lag (T2 position after T1)")
    ax[1].set_ylabel(r"P(T2 correct $\mid$ T1 correct)")
    ax[1].set_ylim(0.4, 0.98)
    ax[1].axvspan(2.5, 4.5, color="#7048E8", alpha=0.08, lw=0)
    ax[1].text(3.5, 0.955, "where cost lands", ha="center", fontsize=7,
               color="#7048E8")
    ax[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    ax[1].set_title("B  A wider blink, not a deeper one", loc="left", fontsize=9)

    fig.tight_layout()
    fig.savefig("figures/fig1_model.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def figure2(story="Growing a plant", lag=4):
    """Per-frame predictions: each account predicts a different shape."""
    rows = [i for i, (s, _f) in enumerate(INDEX) if s == story]
    tr = parse_trace(story, 1.0)
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    x = np.arange(1, len(rows) + 1)
    for a in ["item", "surprisal", "integration"]:
        y = [p_report(lag, PROF[a][i]) for i in rows]
        ax.plot(x, y, "-o", color=C[a], lw=1.3, ms=3.2, label=a + " cost")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n{c}" for i, c in zip(x, tr["cats"])], fontsize=7)
    ax.set_xlabel("frame serving as T1 (number and narrative category)")
    ax.set_ylabel(f"predicted P(T2 | T1)\nat lag {lag}")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower left")
    ax.set_title(f'"{story}"', loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/fig2_profiles.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def figure3():
    """Where on the lag curve the cost signal lives."""
    w = R["where"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(LAGS))
    width = 0.26
    for k, a in enumerate(["item", "surprisal", "integration"]):
        ax.bar(x + (k - 1) * width, [w[a][str(l)] for l in LAGS], width,
               color=C[a], label=a)
    ax.set_xticks(x); ax.set_xticklabels(LAGS)
    ax.set_xlabel("lag")
    ax.set_ylabel("range of predicted\nT2 accuracy across frames")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig("figures/fig3_where.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def figure4():
    """Power of three designs at a fixed trial budget."""
    p = R["design"]["power"]
    ns = ["20", "35", "60"]
    labels = [("binary_trough", "two conditions,\ntrough contrast", "#ADB5BD"),
              ("binary_width", "two conditions,\nwidth contrast", "#748FFC"),
              ("parametric", "parametric\nT1 sampling", "#7048E8")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for ax, a in zip(axes, ["item", "surprisal", "integration"]):
        x = np.arange(len(ns))
        for k, (key, lab, col) in enumerate(labels):
            ax.bar(x + (k - 1) * 0.27, [p[a][n][key] for n in ns], 0.27,
                   color=col, label=lab if a == "item" else None)
        ax.axhline(0.9, color="k", lw=0.6, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(["n=20", "n=35", "n=60"])
        ax.set_title(f"{a} cost", fontsize=9)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("power")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig("figures/fig4_power.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure1(); figure2(); figure3(); figure4()
    print("figures written to figures/")
