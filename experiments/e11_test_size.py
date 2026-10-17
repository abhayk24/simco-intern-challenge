"""Run 11: how big must the real US shopper test be?

Proposed test (online only): each shopper is randomly shown one of 4 versions:
    price $3.49 or $4.49   x   message "environment" or "cleaning"
low_odor is dropped (best in only 1% of refits, run 5). $5.49 is dropped (lowest share, and it
only wins on profit if a bottle costs over ~$2.36).

Shares come straight from run 5 (model D, online). The standard two-group formula
(5% false alarm, 80% chance to detect) gives shoppers needed per group.
Real shoppers may react less than this fictional data, so I also size for HALF the gap.
"""

from math import ceil

from runlog import get_run, run_experiment

EXPECTATION = """
Price ($3.49 vs $4.49, about 8-9 points): about 400 shoppers per price group.
Message (environment vs cleaning, about 4 points): about 1,800-2,000 per message group.
At half the gap, both need about 4 times more.
So message, not price, decides how big the test must be.
"""

INPUTS = {"shares_from_run": 5, "alpha_two_sided": 0.05, "power": 0.8, "channel": "online"}

Z = 1.959964 + 0.841621  # 5% two-sided, 80% power


def per_group(p1, p2):
    return ceil(Z ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2)


def experiment(inputs):
    options = get_run(inputs["shares_from_run"])["outputs"]["options"]
    cell = {(p, m): options[f"{p} / {m} / online"] for p in ("3.49", "4.49") for m in ("environment", "cleaning")}
    share = {k: v["entrant_share"] for k, v in cell.items()}
    avg = lambda keys: sum(share[k] for k in keys) / len(keys)
    price = (avg([("3.49", "environment"), ("3.49", "cleaning")]), avg([("4.49", "environment"), ("4.49", "cleaning")]))
    message = (avg([("3.49", "environment"), ("4.49", "environment")]), avg([("3.49", "cleaning"), ("4.49", "cleaning")]))

    def half(pair):
        mid = sum(pair) / 2
        gap = (pair[0] - pair[1]) / 2
        return mid + gap / 2, mid - gap / 2

    out = {"predicted_cells": {f"{p} / {m}": {"share": v["entrant_share"], "range_90": v["share_90"]} for (p, m), v in cell.items()}}
    for name, pair in (("price_3.49_vs_4.49", price), ("message_environment_vs_cleaning", message)):
        full, halved = per_group(*pair), per_group(*half(pair))
        out[name] = {"shares": [round(x, 3) for x in pair], "gap_points": round(100 * (pair[0] - pair[1]), 1),
                     "shoppers_per_group": full, "shoppers_per_group_if_half_the_gap": halved,
                     "total_test_shoppers": 2 * full, "total_if_half_the_gap": 2 * halved}
    # In a 2x2 test every shopper counts for both questions, so the bigger need sets the size.
    need = max(out["message_environment_vs_cleaning"]["total_test_shoppers"], out["price_3.49_vs_4.49"]["total_test_shoppers"])
    need_half = max(out["message_environment_vs_cleaning"]["total_if_half_the_gap"], out["price_3.49_vs_4.49"]["total_if_half_the_gap"])
    out["two_by_two_total_shoppers"] = {"model_gaps": need, "half_gaps": need_half,
                                        "per_version_model_gaps": ceil(need / 4), "per_version_half_gaps": ceil(need_half / 4)}
    return out


if __name__ == "__main__":
    run_id, out = run_experiment("e11 size the real US shopper test", EXPECTATION, INPUTS, "sizing-v1", experiment)
    for k, v in out.items():
        print(k, v)
