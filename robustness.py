"""
robustness.py
-------------
Three robustness analyses requested at review.

  R1  Coding reliability. Does disagreement between coders destroy the ability
      to recover the generating cost function?
  R2  Functional form. Is the multiplicative integration cost distinguishable
      from an additive variant?
  R3  Trough depth. Do the results survive a deeper baseline blink, closer to
      the deeper end of the published range?

Usage:  python3 robustness.py    (writes robustness.json)
"""

import json
import numpy as np
from scipy import stats

import model
from model import STORIES, parse_trace, p_report, lag_curve

LAGS = [1, 2, 3, 4, 5, 6, 8]
WIDTH = [3, 4]                      # indices of lags 4 and 5
ACCOUNTS = ["item", "surprisal", "integration"]
SD_INTERCEPT, SD_SLOPE = 0.5, 0.35

logit = lambda p: np.log(p / (1 - p))
invlogit = lambda x: 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------------
# Coding perturbation
# ---------------------------------------------------------------------------
def perturbed_profiles(rng, p_flip=0.25, integration="multiplicative"):
    """One coder's version of the corpus. Each frame's slot count and link count
    is independently shifted by +/-1 with probability p_flip."""
    raw = {k: [] for k in ACCOUNTS}
    for s in STORIES:
        d = STORIES[s]
        slots = np.asarray(d["slots"], float).copy()
        links = np.asarray(d["links"], float).copy()
        n = len(slots)
        fs = rng.random(n) < p_flip
        slots[fs] += rng.choice([-1.0, 1.0], fs.sum())
        slots = np.clip(slots, 1, None)
        fl = rng.random(n) < p_flip
        links[fl] += rng.choice([-1.0, 1.0], fl.sum())

        tr = parse_trace(s, 1.0)                 # categories unperturbed
        open_before = np.empty(n)
        run = 0.0
        for i in range(n):
            open_before[i] = max(0.0, run)
            run += links[i]
        resolved = np.maximum(0.0, -links)

        raw["item"].append(slots)
        raw["surprisal"].append(tr["surprisal"])
        if integration == "multiplicative":
            raw["integration"].append(resolved * (1 + 0.5 * open_before))
        else:
            raw["integration"].append(resolved + 0.5 * open_before)
    out = {}
    for k, v in raw.items():
        v = np.concatenate(v).astype(float)
        out[k] = (v - v.mean()) / (v.std() + 1e-12)
    return out


def tables(prof, params=None):
    p = params or model.PARAMS
    tab = {a: np.array([[p_report(l, c, p) for c in prof[a]] for l in LAGS])
           for a in ACCOUNTS}
    base = np.array([p_report(l, 0.0, p) for l in LAGS])
    return tab, base


def recovery(gen_prof, ana_prof, n=35, trials=300, t1=0.75, nsim=150,
             params=None, seed=9):
    """Generate under gen_prof, analyse with ana_prof, report confusion."""
    tab, base = tables(gen_prof, params)
    conf = {}
    for true in ACCOUNTS:
        rng = np.random.default_rng(seed)
        wins = {a: 0 for a in ACCOUNTS}
        T = tab[true]
        for _ in range(nsim):
            F, Y = [], []
            for _j in range(n):
                u = rng.normal(0, SD_INTERCEPT)
                m = max(0.0, rng.normal(1.0, SD_SLOPE))
                k = rng.binomial(trials, t1)
                f = rng.integers(0, gen_prof[true].size, k)
                li = np.asarray(WIDTH)[rng.integers(0, 2, k)]
                eta = logit(base[li]) + m * (logit(T[li, f]) - logit(base[li])) + u
                Y.append((rng.random(k) < invlogit(eta)).astype(float))
                F.append(f)
            f, y = np.concatenate(F), np.concatenate(Y)
            best = None
            for a in ACCOUNTS:
                x = ana_prof[a][f]
                b = np.polyfit(x, y, 1)
                pred = np.clip(b[1] + b[0] * x, 1e-3, 1 - 1e-3)
                ll = float(np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)))
                if best is None or ll > best[0]:
                    best = (ll, a)
            wins[best[1]] += 1
        conf[true] = {a: round(v / nsim, 3) for a, v in wins.items()}
    return conf


# ---------------------------------------------------------------------------
def R1_coding_reliability(p_flips=(0.10, 0.25, 0.40), n_coders=12, seed=3):
    """How much do coders agree, and does disagreement break recovery?"""
    out = {}
    for pf in p_flips:
        rng = np.random.default_rng(seed)
        coders = [perturbed_profiles(rng, pf) for _ in range(n_coders)]
        agree = {a: [] for a in ACCOUNTS}
        for i in range(n_coders):
            for j in range(i + 1, n_coders):
                for a in ACCOUNTS:
                    agree[a].append(np.corrcoef(coders[i][a], coders[j][a])[0, 1])
        rng2 = np.random.default_rng(seed + 100)
        gen = perturbed_profiles(rng2, pf)
        ana = perturbed_profiles(rng2, pf)          # a different coder analyses
        conf = recovery(gen, ana)
        out[str(pf)] = dict(
            mean_pairwise_r={a: round(float(np.mean(v)), 3) for a, v in agree.items()},
            recovery_diagonal={a: conf[a][a] for a in ACCOUNTS},
            confusion=conf)
    return out


def R2_functional_form(seed=5):
    """Multiplicative vs additive integration cost."""
    rng = np.random.default_rng(seed)
    mult = perturbed_profiles(rng, 0.0, "multiplicative")
    rng = np.random.default_rng(seed)
    add = perturbed_profiles(rng, 0.0, "additive")
    r = float(np.corrcoef(mult["integration"], add["integration"])[0, 1])

    # can a design tell them apart? treat them as two rival accounts
    tab_m, base = tables(mult)
    rngs = np.random.default_rng(11)
    wins = {"multiplicative": 0, "additive": 0}
    for _ in range(150):
        F, Y = [], []
        for _j in range(35):
            u = rngs.normal(0, SD_INTERCEPT)
            m = max(0.0, rngs.normal(1.0, SD_SLOPE))
            k = rngs.binomial(300, 0.75)
            f = rngs.integers(0, mult["integration"].size, k)
            li = np.asarray(WIDTH)[rngs.integers(0, 2, k)]
            T = tab_m["integration"]
            eta = logit(base[li]) + m * (logit(T[li, f]) - logit(base[li])) + u
            Y.append((rngs.random(k) < invlogit(eta)).astype(float))
            F.append(f)
        f, y = np.concatenate(F), np.concatenate(Y)
        lls = {}
        for name, prof in [("multiplicative", mult["integration"]),
                           ("additive", add["integration"])]:
            x = prof[f]
            b = np.polyfit(x, y, 1)
            pred = np.clip(b[1] + b[0] * x, 1e-3, 1 - 1e-3)
            lls[name] = float(np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)))
        wins[max(lls, key=lls.get)] += 1
    return dict(correlation_between_forms=round(r, 3),
                recovered_as_multiplicative=round(wins["multiplicative"] / 150, 3))


def R3_deeper_trough():
    """Re-run the key results with a baseline trough near 0.35."""
    deep = dict(model.PARAMS)
    deep["block"] = 0.97
    deep["b0"], deep["b1"] = -0.706, 2.903      # full uptake -> .90, full block -> .35

    curve = [round(p_report(l, 0.0, deep), 3) for l in LAGS]
    rng = np.random.default_rng(1)
    prof = perturbed_profiles(rng, 0.0)
    where = {a: {str(l): round(float(np.ptp([p_report(l, c, deep) for c in prof[a]])), 3)
                 for l in LAGS} for a in ACCOUNTS}
    conf = recovery(prof, prof, params=deep)
    return dict(baseline_curve=dict(zip(map(str, LAGS), curve)),
                where=where,
                recovery_diagonal={a: conf[a][a] for a in ACCOUNTS})


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    R = {}
    print("R1  coding reliability")
    R["coding"] = R1_coding_reliability()
    for pf, d in R["coding"].items():
        print(f"   flip={pf}  mean pairwise r={d['mean_pairwise_r']}  "
              f"recovery={d['recovery_diagonal']}")

    print("R2  functional form of integration cost")
    R["form"] = R2_functional_form()
    print("   ", R["form"])

    print("R3  deeper baseline trough")
    R["deep_trough"] = R3_deeper_trough()
    print("    curve:", R["deep_trough"]["baseline_curve"])
    print("    where:", R["deep_trough"]["where"]["integration"])
    print("    recovery:", R["deep_trough"]["recovery_diagonal"])

    with open("robustness.json", "w") as fh:
        json.dump(R, fh, indent=2)
    print("\nwritten: robustness.json")
