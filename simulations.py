"""
simulations.py
--------------
Runs every numerical result reported in the paper and writes results.json.

  Sim 0  Baseline lag curve: does the model produce a normal blink?
  Sim 1  Where on the lag curve does commitment cost show up?
  Sim 2  Are the three cost accounts separable across the stimulus corpus?
  Sim 3  Design comparison at a fixed trial budget
  Sim 4  Trial budget and T1 report accuracy
  Sim 5  Model identifiability
  Sim 6  Sensitivity to the free scaling parameter beta
  Sim 7  Per-frame predicted profiles for one story (for Figure 2)

Usage:  python3 simulations.py
Runtime: about ten minutes on a laptop.
"""

import itertools
import json
import numpy as np
from scipy import stats

import model
from model import cost_profiles, lag_curve, p_report, parse_trace

LAGS = [1, 2, 3, 4, 5, 6, 8]
IDX = {l: i for i, l in enumerate(LAGS)}
ACCOUNTS = ["item", "surprisal", "integration"]

TROUGH = [IDX[2], IDX[3]]        # conventional blink trough
WIDTH = [IDX[4], IDX[5]]         # recovery limb, where a duration effect lands
FLANK = [IDX[1], IDX[8]]

SD_INTERCEPT = 0.5               # between-participant variation in overall accuracy
SD_SLOPE = 0.35                  # between-participant variation in effect size

logit = lambda p: np.log(p / (1.0 - p))
invlogit = lambda x: 1.0 / (1.0 + np.exp(-x))

PROF, INDEX = cost_profiles(1.0)


def tables(prof):
    tab = {a: np.array([[p_report(l, c) for c in prof[a]] for l in LAGS])
           for a in ACCOUNTS}
    base = np.array([p_report(l, 0.0) for l in LAGS])
    return tab, base


# ---------------------------------------------------------------------------
def sim0_baseline():
    return dict(lags=LAGS, accuracy=np.round(lag_curve(LAGS, 0.0), 3).tolist())


def sim1_where_on_the_curve():
    out = {}
    for a in ACCOUNTS:
        out[a] = {str(l): round(float(np.ptp([p_report(l, c) for c in PROF[a]])), 3)
                  for l in LAGS}
    return out


def sim2_separability():
    corr = {f"{a}_vs_{b}": round(float(np.corrcoef(PROF[a], PROF[b])[0, 1]), 3)
            for a, b in itertools.combinations(ACCOUNTS, 2)}
    return dict(n_frames=len(INDEX), correlations=corr)


# ---------------------------------------------------------------------------
def design_parametric(n, account, trials, t1_acc, rng, tab, base,
                      lags_used=WIDTH):
    """T1 is a randomly drawn frame on every trial. Statistic: regression slope."""
    cost = PROF[account]
    T = tab[account]
    out = np.empty(n)
    for j in range(n):
        u = rng.normal(0.0, SD_INTERCEPT)
        m = max(0.0, rng.normal(1.0, SD_SLOPE))
        k = rng.binomial(trials, t1_acc)
        frame = rng.integers(0, cost.size, k)
        li = np.asarray(lags_used)[rng.integers(0, len(lags_used), k)]
        eta = logit(base[li]) + m * (logit(T[li, frame]) - logit(base[li])) + u
        y = (rng.random(k) < invlogit(eta)).astype(float)
        out[j] = np.polyfit(cost[frame], y, 1)[0]
    return out


def design_binary(n, account, trials, t1_acc, rng, tab, base, focus=TROUGH):
    """Two conditions (high-cost vs average-cost T1) crossed with lag.
    Statistic: focus-minus-flank contrast."""
    cost = PROF[account]
    hi = int(np.argmax(cost))
    T = tab[account]
    per_cell = max(1, trials // (2 * len(LAGS)))
    out = np.empty(n)
    for j in range(n):
        u = rng.normal(0.0, SD_INTERCEPT)
        m = max(0.0, rng.normal(1.0, SD_SLOPE))
        d = np.empty(len(LAGS))
        for li in range(len(LAGS)):
            props = []
            for p_true in (T[li, hi], base[li]):
                eta = logit(base[li]) + m * (logit(p_true) - logit(base[li])) + u
                k = max(1, rng.binomial(per_cell, t1_acc))
                props.append(rng.binomial(k, invlogit(eta)) / k)
            d[li] = props[0] - props[1]
        out[j] = d[focus].mean() - d[FLANK].mean()
    return out


def power(design, n, account, trials, t1_acc, tab, base, nsim=300, seed=4, **kw):
    rng = np.random.default_rng(seed)
    hits = sum(stats.ttest_1samp(
        design(n, account, trials, t1_acc, rng, tab, base, **kw), 0.0)[1] < 0.05
        for _ in range(nsim))
    return round(hits / nsim, 3)


def sim3_design_comparison(trials=400, t1_acc=0.75, ns=(20, 35, 60)):
    tab, base = tables(PROF)
    res = {}
    for a in ACCOUNTS:
        res[a] = {}
        for n in ns:
            res[a][str(n)] = dict(
                binary_trough=power(design_binary, n, a, trials, t1_acc, tab,
                                    base, focus=TROUGH),
                binary_width=power(design_binary, n, a, trials, t1_acc, tab,
                                   base, focus=WIDTH),
                parametric=power(design_parametric, n, a, trials, t1_acc, tab, base))
    return dict(trials=trials, t1_accuracy=t1_acc, power=res)


def sim4_budget(account="integration", t1_acc=0.75,
                budgets=(50, 100, 150, 250, 400), ns=(12, 20, 35)):
    tab, base = tables(PROF)
    return {str(b): {str(n): power(design_parametric, n, account, b, t1_acc,
                                   tab, base, nsim=200)
                     for n in ns} for b in budgets}


def sim4b_t1(account="integration", trials=150, n=20,
             t1_levels=(0.48, 0.60, 0.75, 0.90)):
    tab, base = tables(PROF)
    return {str(t): power(design_parametric, n, account, trials, t, tab, base,
                          nsim=200) for t in t1_levels}


# ---------------------------------------------------------------------------
def sim5_identifiability(n=35, trials=300, t1_acc=0.75, nsim=200, seed=9):
    tab, base = tables(PROF)
    conf = {}
    for true in ACCOUNTS:
        rng = np.random.default_rng(seed)
        wins = {a: 0 for a in ACCOUNTS}
        T = tab[true]
        for _ in range(nsim):
            frames, ys = [], []
            for _j in range(n):
                u = rng.normal(0.0, SD_INTERCEPT)
                m = max(0.0, rng.normal(1.0, SD_SLOPE))
                k = rng.binomial(trials, t1_acc)
                f = rng.integers(0, PROF[true].size, k)
                li = np.asarray(WIDTH)[rng.integers(0, len(WIDTH), k)]
                eta = logit(base[li]) + m * (logit(T[li, f]) - logit(base[li])) + u
                ys.append((rng.random(k) < invlogit(eta)).astype(float))
                frames.append(f)
            f, y = np.concatenate(frames), np.concatenate(ys)
            best = None
            for a in ACCOUNTS:
                x = PROF[a][f]
                b = np.polyfit(x, y, 1)
                pred = np.clip(b[1] + b[0] * x, 1e-3, 1 - 1e-3)
                ll = float(np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)))
                if best is None or ll > best[0]:
                    best = (ll, a)
            wins[best[1]] += 1
        conf[true] = {a: round(v / nsim, 3) for a, v in wins.items()}
    return dict(n=n, trials=trials, confusion=conf)


# ---------------------------------------------------------------------------
def sim6_beta(betas=(6.0, 12.0, 24.0), n=20, trials=150, t1_acc=0.75):
    global PROF, INDEX
    out, original = {}, model.PARAMS["beta"]
    for b in betas:
        model.PARAMS["beta"] = b
        PROF, INDEX = cost_profiles(1.0)
        tab, base = tables(PROF)
        out[str(b)] = dict(
            power=power(design_parametric, n, "integration", trials, t1_acc,
                        tab, base, nsim=200),
            max_effect_lag5=round(float(np.ptp([p_report(5, c)
                                                for c in PROF["integration"]])), 3))
    model.PARAMS["beta"] = original
    PROF, INDEX = cost_profiles(1.0)
    return out


def sim7_frame_profiles(story="Growing a plant", lag=5):
    rows = [i for i, (s, _f) in enumerate(INDEX) if s == story]
    tr = parse_trace(story, 1.0)
    return dict(story=story, lag=lag, categories=tr["cats"],
                open_before=tr["open_before"].tolist(),
                predicted={a: np.round([p_report(lag, PROF[a][i]) for i in rows],
                                       3).tolist() for a in ACCOUNTS})


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    R = {}
    print("Sim 0  baseline lag curve")
    R["baseline"] = sim0_baseline(); print("  ", R["baseline"]["accuracy"])

    print("Sim 1  where cost shows up on the curve")
    R["where"] = sim1_where_on_the_curve(); print("  ", R["where"]["integration"])

    print("Sim 2  separability of cost accounts")
    R["separability"] = sim2_separability(); print("  ", R["separability"]["correlations"])

    print("Sim 3  design comparison")
    R["design"] = sim3_design_comparison()
    for a, d in R["design"]["power"].items():
        print(f"   {a:12s}", {k: tuple(v.values()) for k, v in d.items()})

    print("Sim 4  trial budget")
    R["budget"] = sim4_budget(); print("  ", R["budget"])

    print("Sim 4b T1 report accuracy")
    R["t1"] = sim4b_t1(); print("  ", R["t1"])

    print("Sim 5  identifiability")
    R["identifiability"] = sim5_identifiability()
    for k, v in R["identifiability"]["confusion"].items():
        print(f"   true={k:12s}", v)

    print("Sim 6  beta sensitivity")
    R["beta"] = sim6_beta(); print("  ", R["beta"])

    print("Sim 7  per-frame profiles")
    R["profiles"] = sim7_frame_profiles()

    with open("results.json", "w") as fh:
        json.dump(R, fh, indent=2)
    print("\nwritten: results.json")
