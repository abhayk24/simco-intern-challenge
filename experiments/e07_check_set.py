"""Run 7: the one-time test on the check set (50 households, 300 visits, never used before).

Model D is locked: same pieces and settings as runs 4 and 5, trained on ALL build visits.
Nothing is tuned on the check set. If I change the model after this, I will say so and
I will not call a later check-set result a fresh test.
"""

import math
import random
from collections import defaultdict

from baseline import predict as starter_predict
from choice_model import ChoiceModel
from experiments.common import by_household, load, score
from run_baseline import offers_for
from runlog import run_experiment

EXPECTATION = """
PASS RULES, decided now:
 (1) D log loss on check is lower than the starter (1.5626) AND the constant model (1.5905).
 (2) D's predicted entrant share is within 3 points of the real check share (18.3%).
 (3) Raw check data: entrant share at $3.49 is above $5.49.
Message and channel directions on check are reported but cannot pass or fail anything:
with about 100 visits per message, they are too noisy to settle it.
My guesses: D about 1.555, beating the starter by less than 0.01, and the household
interval for "D minus starter" probably includes zero. Entrant share predicted about 20%.
Price direction holds. environment > low_odor and online > retail: I think they hold, but weakly.
"""

INPUTS = {"train": "visits.csv (all 200 households)", "test": "check_visits.csv (50 households)",
          "model": "D = brand, price, household, levers, history", "ridge": 0.01, "reps": 2000, "seed": 3}

GROUPS = ("brand", "price", "household", "levers", "history")


def per_visit_loss(predict, visits, households):
    history, out = defaultdict(list), []
    for v in visits:
        hid = v["household_id"]
        p = predict(households[hid], offers_for(v), v, history[hid])["probabilities"]
        out.append((hid, -math.log(max(p[v["choice"]], 1e-12)), p["entrant"]))
        history[hid].append(v)
    return out


def raw_share(visits, column, level):
    rows = [v for v in visits if v[column] == level]
    return round(sum(v["choice"] == "entrant" for v in rows) / len(rows), 3), len(rows)


def experiment(inputs):
    households, build, check = load()
    model = ChoiceModel(GROUPS, inputs["ridge"]).fit(build, households, offers_for)
    model_c = ChoiceModel(GROUPS[:-1], inputs["ridge"]).fit(build, households, offers_for)
    shares = {o: sum(v["choice"] == o for v in build) / len(build) for o in ("entrant", "standard", "budget", "scented", "none")}
    constant = lambda *a: {"probabilities": shares}

    out = {name: score(fn, check, households) for name, fn in
           [("D", model.predict), ("C_no_history", model_c.predict), ("starter", starter_predict), ("constant", constant)]}

    # how sure is "D beats starter"? resample check households
    d = per_visit_loss(model.predict, check, households)
    s = per_visit_loss(starter_predict, check, households)
    diff_by_h = defaultdict(list)
    for (hid, ld, _), (_, ls, _) in zip(d, s):
        diff_by_h[hid].append(ld - ls)
    groups = list(diff_by_h.values())
    rng = random.Random(inputs["seed"])
    means = sorted(sum(sum(g) for g in sample) / sum(len(g) for g in sample)
                   for sample in (rng.choices(groups, k=len(groups)) for _ in range(inputs["reps"])))
    out["D_minus_starter_log_loss"] = {"point": round(sum(sum(g) for g in groups) / len(check), 4),
                                       "household_95": [round(means[int(0.025 * len(means))], 4),
                                                        round(means[int(0.975 * len(means))], 4)]}

    # calibration: do visits where D says "entrant likely" really buy it more?
    bins = defaultdict(lambda: [0, 0.0, 0])
    for v, (_, _, pe) in zip(check, d):
        b = "<15%" if pe < 0.15 else "15-25%" if pe < 0.25 else "25%+"
        bins[b][0] += 1; bins[b][1] += pe; bins[b][2] += v["choice"] == "entrant"
    out["entrant_calibration"] = {b: {"visits": n, "predicted": round(sp / n, 3), "actual": round(a / n, 3)}
                                  for b, (n, sp, a) in sorted(bins.items())}

    out["raw_check_directions"] = {
        "price": {p: raw_share(check, "entrant_price", p) for p in ("3.49", "4.49", "5.49")},
        "message": {m: raw_share(check, "entrant_message", m) for m in ("cleaning", "environment", "low_odor")},
        "channel": {c: raw_share(check, "channel", c) for c in ("online", "retail")},
    }
    out["rules"] = {
        "1_beats_starter_and_constant": out["D"]["log_loss"] < out["starter"]["log_loss"] and out["D"]["log_loss"] < out["constant"]["log_loss"],
        "2_entrant_share_within_3_points": abs(out["D"]["share_expected"]["entrant"] - out["D"]["share_observed"]["entrant"]) <= 0.03,
        "3_price_direction_holds": out["raw_check_directions"]["price"]["3.49"][0] > out["raw_check_directions"]["price"]["5.49"][0],
    }
    return out


if __name__ == "__main__":
    run_id, out = run_experiment("e07 one-time check set test of locked model D", EXPECTATION, INPUTS,
                                 "logit-v2-history", experiment)
    for k, v in out.items():
        print(k, v)
