"""
model.py
--------
The attentional blink as a serial-commitment bottleneck.

Three parts:
  1. A stimulus corpus of visual narratives, coded as event grammars.
  2. A parse-state trace for each story, giving per-frame commitment cost
     under three rival cost functions.
  3. A commitment model that turns cost into a predicted T2 report probability.

The model is stated at the computational level. It says nothing about which
neural circuit implements the suspension of uptake.

Rakesh Sengupta & Aahana Rath, Computational Cognition Lab, Krea University.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. STIMULUS CORPUS
# ---------------------------------------------------------------------------
# Visual narrative grammar categories (after Cohn, 2013).
CATS = ["E", "I", "L", "P", "R"]     # Establisher, Initial, Prolongation, Peak, Release
CIX = {c: i for i, c in enumerate(CATS)}

# Canonical arc affinities: how strongly each category predicts the next.
AFFINITY = np.array([
    #  to:   E     I     L     P     R
    [0.15, 0.60, 0.20, 0.03, 0.02],   # from E
    [0.02, 0.15, 0.55, 0.25, 0.03],   # from I
    [0.02, 0.10, 0.40, 0.45, 0.03],   # from L
    [0.02, 0.05, 0.08, 0.15, 0.70],   # from P
    [0.30, 0.25, 0.15, 0.05, 0.25],   # from R
])


def transition_matrix(gamma=1.0):
    """Grammar knowledge. gamma = 0 is a flat prior, gamma = 1 the full grammar."""
    flat = np.ones_like(AFFINITY) / len(CATS)
    M = (1.0 - gamma) * flat + gamma * AFFINITY
    return M / M.sum(axis=1, keepdims=True)


# Four 18-frame picture stories.
#   cats  : narrative category of each frame
#   slots : count of depicted entities and relations (item-intrinsic load)
#   links : dependencies opened (+) or resolved (-) at each frame
STORIES = {
    "An evening in": dict(
        cats="E E I L L L P L L R I L L P L R R R",
        slots=[2, 2, 3, 3, 4, 4, 5, 4, 3, 3, 3, 4, 4, 5, 3, 3, 2, 2],
        links=[1, 1, 2, 1, 1, 1, 0, 1, 1, -2, 2, 1, 1, 0, 1, -2, -1, -1]),
    "Growing a plant": dict(
        cats="E I L L L P L L R L L P L R R R E R",
        slots=[2, 3, 3, 3, 3, 4, 3, 3, 3, 3, 4, 5, 3, 3, 2, 2, 2, 2],
        links=[1, 2, 1, 1, 1, 0, 1, 1, -2, 1, 1, 0, 1, -1, -1, -1, 1, -1]),
    "Posting a letter": dict(
        cats="E I I L L P R L I L P L L R R L R R",
        slots=[2, 3, 3, 3, 4, 5, 3, 3, 3, 4, 5, 3, 3, 3, 2, 3, 2, 2],
        links=[1, 2, 1, 1, 1, 0, -2, 1, 2, 1, 0, 1, 1, -2, -1, 1, -1, -1]),
    "Getting to school": dict(
        cats="E E I L P L L R I L L L P L R R R R",
        slots=[2, 2, 3, 3, 5, 3, 3, 3, 3, 4, 4, 3, 5, 3, 3, 2, 2, 2],
        links=[1, 1, 2, 1, 0, 1, 1, -2, 2, 1, 1, 1, 0, 1, -2, -1, -1, -1]),
}

N_FRAMES = 18


# ---------------------------------------------------------------------------
# 2. PARSE STATE AND COST
# ---------------------------------------------------------------------------
def parse_trace(story, gamma=1.0):
    """Per-frame parse state for one story."""
    d = STORIES[story]
    cats = d["cats"].split()
    slots = np.asarray(d["slots"], float)
    links = np.asarray(d["links"], float)
    n = len(cats)

    M = transition_matrix(gamma)
    surprisal = np.empty(n)
    for i in range(n):
        p = M[CIX[cats[i - 1]], CIX[cats[i]]] if i > 0 else 1.0 / len(CATS)
        surprisal[i] = -np.log2(p)

    open_before = np.empty(n)
    running = 0.0
    for i in range(n):
        open_before[i] = max(0.0, running)
        running += links[i]
    resolved = np.maximum(0.0, -links)

    return dict(cats=cats, slots=slots, surprisal=surprisal,
                open_before=open_before, resolved=resolved)


def cost_item(tr):
    """Item-intrinsic load. How much is depicted in this frame."""
    return tr["slots"]


def cost_surprisal(tr):
    """Predictive load. How poorly the preceding frames predicted this one."""
    return tr["surprisal"]


def cost_integration(tr):
    """Structural load. How much pending structure this frame closes."""
    return tr["resolved"] * (1.0 + 0.5 * tr["open_before"])


COSTS = {"item": cost_item, "surprisal": cost_surprisal, "integration": cost_integration}


def cost_profiles(gamma=1.0):
    """Cost of every frame in the corpus, z-scored within each account.

    Returns (profiles, index) where index[k] = (story, frame) for column k.
    """
    raw = {k: [] for k in COSTS}
    index = []
    for s in STORIES:
        tr = parse_trace(s, gamma)
        for k, f in COSTS.items():
            raw[k].append(f(tr))
        index += [(s, i) for i in range(len(tr["cats"]))]
    out = {}
    for k, v in raw.items():
        v = np.concatenate(v).astype(float)
        out[k] = (v - v.mean()) / (v.std() + 1e-12)
    return out, index


# ---------------------------------------------------------------------------
# 3. COMMITMENT MODEL
# ---------------------------------------------------------------------------
PARAMS = dict(
    t_start=270.0,   # ms after T1 onset at which commitment begins
    dur0=260.0,      # ms of commitment for a frame of average cost
    beta=12.0,       # ms of extra commitment per z-unit of cost
    block=0.90,      # fraction of uptake suspended during commitment
    tau=25.0,        # ms, softness of the interval edges
    window=200.0,    # ms of evidence T2 needs
    b0=-0.388,       # evidence-to-accuracy link
    b1=2.585,
)


def availability(t, cost_z, p=PARAMS):
    """Fraction of uptake capacity available at time t after T1 onset."""
    t0 = p["t_start"]
    t1 = t0 + p["dur0"] + p["beta"] * cost_z
    on = 1.0 / (1.0 + np.exp(-(t - t0) / p["tau"]))
    off = 1.0 / (1.0 + np.exp((t - t1) / p["tau"]))
    return 1.0 - p["block"] * on * off


def p_report(lag, cost_z, p=PARAMS, soa=100.0, grid=400):
    """Probability of reporting T2, given T1 was reported."""
    t0 = lag * soa
    t = np.linspace(t0, t0 + p["window"], grid)
    evidence = float(np.trapezoid(availability(t, cost_z, p), t) / p["window"])
    return float(1.0 / (1.0 + np.exp(-(p["b0"] + p["b1"] * evidence))))


def lag_curve(lags, cost_z=0.0, p=PARAMS):
    return np.array([p_report(l, cost_z, p) for l in lags])
