# narrative-blink

Simulation code for **"What visual narrative adds to the attentional blink: a
parsing account and its consequences for design"**.

Aahana Rath and Rakesh Sengupta
Computational Cognition Lab, School of Interwoven Arts and Sciences,
Krea University, Sri City, India

---

## What this is

A computational-level account of the attentional blink as a serial-commitment
bottleneck. Any system that parses a stream must commit to one unit before it can
take up the next; commitment takes time; uptake is suspended while it lasts. The
question that follows is what makes committing to one item cost more than
committing to another, and visual narrative is used here to make that cost
computable frame by frame.

Nothing in this repository requires human data. Every number in the paper is
produced by the scripts below.

## Repository contents

| File | Purpose |
|---|---|
| `model.py` | Stimulus corpus coded as event grammars, parse-state trace, three cost functions, commitment model |
| `simulations.py` | Runs all seven simulations, writes `results.json` |
| `figures.py` | Builds the four figures from `results.json` |
| `export_corpus.py` | Writes `corpus.csv`, the full frame-by-frame stimulus coding |
| `corpus.csv` | Stimulus coding and derived cost values, for inspection or re-coding |
| `results.json` | Numerical output; every value in the paper comes from here |
| `requirements.txt`, `LICENSE`, `CITATION.cff` | Environment, licence, citation metadata |

## Reproducing everything

```bash
pip install -r requirements.txt
python3 simulations.py      # ~10 minutes, writes results.json
python3 export_corpus.py    # writes corpus.csv
python3 figures.py          # writes figures/*.pdf
```

## What each simulation shows

| Simulation | Result |
|---|---|
| 0 | The model produces a canonical blink at zero cost: lag-1 sparing, trough at lag 3, recovery by lag 6 |
| 1 | Commitment cost appears at lags 4 and 5, not at the trough — the blink widens rather than deepens |
| 2 | The three cost functions correlate weakly across the 72 frames (r between −0.33 and +0.61) |
| 3 | Parametric sampling of the first target outperforms the two-condition design; the conventional trough contrast stays at the nominal error rate |
| 4 | Trial count binds harder than sample size, since conditioning on correct T1 discards roughly half the trials |
| 5 | The generating cost function is recovered on ~97% of simulated datasets at n = 35 |
| 6 | The free scaling parameter β changes required sample size but not the qualitative pattern |

## Changing or extending the stimulus corpus

Stories live in `STORIES` in `model.py`. Each needs three coded fields of equal
length:

- `cats` — narrative categories (E/I/L/P/R, after Cohn 2013)
- `slots` — count of depicted entities and relations
- `links` — dependencies opened (positive) or resolved (negative)

Adding stories with more closure points will raise the variance of the
integration cost function, which is currently the weakest of the three. Inspect
`corpus.csv` to see the effect of a re-coding without reading any Python.

## Citation

See `CITATION.cff`. Please cite the paper once it is published; until then, cite
this repository and its archived release.

## Licence

MIT (see `LICENSE`). The stimulus coding in `corpus.csv` may be reused under the
same terms with attribution.
