"""Run 2: the FIRST.md test. How sure are we about the price, message and channel gaps?

Each household has 6 visits, so visits are not independent. I resample whole households
(cluster bootstrap) and, for comparison, single visits (the naive way). If household
resampling gives clearly wider intervals, my FIRST.md concern was justified.

Build set only. The check set is not touched.
"""

import random
from collections import defaultdict

from experiments.common import by_household, load
from runlog import run_experiment

EXPECTATION = """
Price: $3.49 minus $5.49 entrant share interval excludes zero, roughly +10 to +22 points.
Message: environment minus low_odor is borderline; I expect the lower end of the interval near 0.
Cleaning vs the others: interval includes zero.
Channel: online minus retail is borderline, lower end near or below 0.
Household resampling gives wider intervals than visit resampling, by 10-30%.
"""

INPUTS = {"data": "visits.csv (build only)", "reps": 4000, "seed": 7, "interval": "95% percentile"}

# Each contrast: (lever column, level A, level B). Result is share(A) - share(B).
CONTRASTS = [
    ("entrant_price", "3.49", "4.49"),
    ("entrant_price", "4.49", "5.49"),
    ("entrant_price", "3.49", "5.49"),
    ("entrant_message", "environment", "low_odor"),
    ("entrant_message", "environment", "cleaning"),
    ("entrant_message", "cleaning", "low_odor"),
    ("channel", "online", "retail"),
]


def entrant_share_gaps(visits):
    hits = defaultdict(int)
    total = defaultdict(int)
    for v in visits:
        for column in ("entrant_price", "entrant_message", "channel"):
            key = (column, v[column])
            total[key] += 1
            hits[key] += v["choice"] == "entrant"
    share = lambda c, lvl: hits[(c, lvl)] / total[(c, lvl)]
    return {f"{c}: {a} - {b}": share(c, a) - share(c, b) for c, a, b in CONTRASTS}


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def experiment(inputs):
    _, build, _ = load()
    groups = list(by_household(build).values())
    rng = random.Random(inputs["seed"])
    point = entrant_share_gaps(build)
    cluster = defaultdict(list)
    naive = defaultdict(list)
    for _ in range(inputs["reps"]):
        sample = [v for g in rng.choices(groups, k=len(groups)) for v in g]
        for k, gap in entrant_share_gaps(sample).items():
            cluster[k].append(gap)
        for k, gap in entrant_share_gaps(rng.choices(build, k=len(build))).items():
            naive[k].append(gap)
    result = {}
    for k in point:
        lo, hi = percentile(cluster[k], 0.025), percentile(cluster[k], 0.975)
        nlo, nhi = percentile(naive[k], 0.025), percentile(naive[k], 0.975)
        result[k] = {
            "gap_points": round(100 * point[k], 1),
            "household_95": [round(100 * lo, 1), round(100 * hi, 1)],
            "visit_95": [round(100 * nlo, 1), round(100 * nhi, 1)],
            "width_ratio_household_vs_visit": round((hi - lo) / (nhi - nlo), 2),
            "excludes_zero": lo > 0 or hi < 0,
            "share_of_reps_above_zero": round(sum(g > 0 for g in cluster[k]) / len(cluster[k]), 3),
        }
    return result


if __name__ == "__main__":
    run_id, out = run_experiment("e02 household bootstrap of lever gaps", EXPECTATION, INPUTS,
                                 "bootstrap-v1", experiment)
    print(f"{'contrast':42} {'gap':>6}  {'household 95%':>15}  {'visit 95%':>15}  ratio  excl0")
    for k, r in out.items():
        print(f"{k:42} {r['gap_points']:>6}  {str(r['household_95']):>15}  {str(r['visit_95']):>15}  "
              f"{r['width_ratio_household_vs_visit']:>5}  {r['excludes_zero']}")
