"""
export_corpus.py
----------------
Writes corpus.csv: the full frame-by-frame coding of the four stories, together
with the derived parse-state quantities and the three cost values.

This exists so that the stimulus coding can be inspected, checked and re-coded by
someone who does not want to read Python.

Usage:  python3 export_corpus.py
"""

import csv
import numpy as np

from model import STORIES, cost_profiles, parse_trace

FIELDS = ["story", "frame", "category", "slots", "links",
          "open_before", "resolved", "surprisal",
          "cost_item_z", "cost_surprisal_z", "cost_integration_z"]


def main(gamma=1.0, path="corpus.csv"):
    prof, index = cost_profiles(gamma)
    rows = []
    for k, (story, i) in enumerate(index):
        tr = parse_trace(story, gamma)
        rows.append({
            "story": story,
            "frame": i + 1,
            "category": tr["cats"][i],
            "slots": int(STORIES[story]["slots"][i]),
            "links": int(STORIES[story]["links"][i]),
            "open_before": round(float(tr["open_before"][i]), 2),
            "resolved": round(float(tr["resolved"][i]), 2),
            "surprisal": round(float(tr["surprisal"][i]), 3),
            "cost_item_z": round(float(prof["item"][k]), 3),
            "cost_surprisal_z": round(float(prof["surprisal"][k]), 3),
            "cost_integration_z": round(float(prof["integration"][k]), 3),
        })
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"written: {path} ({len(rows)} frames)")


if __name__ == "__main__":
    main()
