"""Run 3: build the learned model one idea at a time and see which ideas really help.

How I judge "helps": 5-fold cross-validation on the BUILD set, split by household.
The model learns on 4/5 of the households and is scored on the 1/5 it never saw.
The check set stays untouched.

Models:
  A  = brand pull + price
  B  = A + household (usual brand, tight / flexible budget changes price feeling)
  B+msg  = B + entrant message
  B+chan = B + entrant online
  C  = B + message + channel
"""

import math
import random

from baseline import predict as starter_predict
from choice_model import ChoiceModel, MESSAGES
from experiments.common import by_household, load, score
from run_baseline import offers_for
from runlog import run_experiment

EXPECTATION = """
(I already fit model A once while testing the code: train log loss 1.58, price weight -0.46.)
A: cross-validated log loss about 1.585, no better than the starter (1.579).
B: usual brand helps clearly, CV log loss about 1.55, usual-brand weight about +0.5.
B+msg and B+chan: each improves CV log loss by less than 0.003, maybe none.

RULE, decided now: I call message or channel "real" only if BOTH
  (1) its Wald test inside model C has p < 0.01 (strict, because I test several things), and
  (2) adding it to B lowers cross-validated log loss.
My guess: message p between 0.01 and 0.1 and channel p about 0.03, so both FAIL the rule.
"""

INPUTS = {"data": "visits.csv (build only)", "folds": 5, "split": "by household", "seed": 11, "ridge": 0.01}

MODELS = {
    "A": ("brand", "price"),
    "B": ("brand", "price", "household"),
    "B+msg": ("brand", "price", "household", "levers:msg"),
    "B+chan": ("brand", "price", "household", "levers:chan"),
    "C": ("brand", "price", "household", "levers"),
}


class Subset(ChoiceModel):
    """Lets B+msg / B+chan use only part of the 'levers' group by dropping the other features."""

    def __init__(self, groups, ridge):
        self.keep = None
        if "levers:msg" in groups:
            self.keep = "msg"
        if "levers:chan" in groups:
            self.keep = "chan"
        groups = tuple(g if not g.startswith("levers") else "levers" for g in groups)
        super().__init__(groups, ridge)

    def _rows(self, offers, household, visit, history):
        rows = super()._rows(offers, household, visit, history)
        if self.keep == "msg":
            rows = [(n, {k: v for k, v in f.items() if k != "entrant_online"}) for n, f in rows]
        if self.keep == "chan":
            rows = [(n, {k: v for k, v in f.items() if not k.startswith("entrant_msg_")}) for n, f in rows]
        return rows


def folds(visits, k, seed):
    ids = sorted(by_household(visits))
    random.Random(seed).shuffle(ids)
    fold_of = {hid: i % k for i, hid in enumerate(ids)}
    return [([v for v in visits if fold_of[v["household_id"]] != f],
             [v for v in visits if fold_of[v["household_id"]] == f]) for f in range(k)]


def cross_validate(groups, visits, households, inputs):
    total_loss = total_hit = n = 0
    for train, test in folds(visits, inputs["folds"], inputs["seed"]):
        model = Subset(groups, inputs["ridge"]).fit(train, households, offers_for)
        s = score(model.predict, test, households)
        total_loss += s["log_loss"] * s["visits"]
        total_hit += s["top_choice_matched"] * s["visits"]
        n += s["visits"]
    return {"cv_log_loss": round(total_loss / n, 4), "cv_top_choice": round(total_hit / n, 4)}


def wald(model, names):
    """Joint test that all listed weights are equal to each other (len 1: equal to zero)."""
    w, cov = model.weights, model.covariance
    if len(names) == 1:
        z = w[names[0]] / math.sqrt(cov[(names[0], names[0])])
        return {"stat": round(z * z, 2), "df": 1, "p": round(math.erfc(abs(z) / math.sqrt(2)), 4)}
    base = names[-1]  # differences against the last one; df = len - 1
    diffs = [n for n in names[:-1]]
    d = [w[a] - w[base] for a in diffs]
    v = [[cov[(a, b)] - cov[(a, base)] - cov[(base, b)] + cov[(base, base)] for b in diffs] for a in diffs]
    det = v[0][0] * v[1][1] - v[0][1] * v[1][0]
    inv = [[v[1][1] / det, -v[0][1] / det], [-v[1][0] / det, v[0][0] / det]]
    stat = sum(d[i] * inv[i][j] * d[j] for i in range(2) for j in range(2))
    return {"stat": round(stat, 2), "df": 2, "p": round(math.exp(-stat / 2), 4)}  # chi2 df=2 tail


def experiment(inputs):
    households, build, _ = load()
    out = {"starter_full_build": score(starter_predict, build, households)["log_loss"]}
    for name, groups in MODELS.items():
        out[name] = cross_validate(groups, build, households, inputs)
    full = Subset(MODELS["C"], inputs["ridge"]).fit(build, households, offers_for)
    out["C_weights"] = {k: [round(full.weights[k], 3), round(full.std_errors[k], 3)] for k in sorted(full.weights)}
    msg = wald(full, [f"entrant_msg_{m}" for m in MESSAGES])
    chan = wald(full, ["entrant_online"])
    out["wald_message_all_equal"] = msg
    out["wald_channel_zero"] = chan
    out["rule"] = {
        "message_real": msg["p"] < 0.01 and out["B+msg"]["cv_log_loss"] < out["B"]["cv_log_loss"],
        "channel_real": chan["p"] < 0.01 and out["B+chan"]["cv_log_loss"] < out["B"]["cv_log_loss"],
    }
    return out


if __name__ == "__main__":
    run_id, out = run_experiment("e03 grow the learned model one idea at a time", EXPECTATION, INPUTS,
                                 "logit-v1", experiment)
    for k, v in out.items():
        if k != "C_weights":
            print(f"{k:24} {v}")
    print("\nModel C weights [value, std error]:")
    for k, v in out["C_weights"].items():
        print(f"  {k:28} {v[0]:+.3f}  ±{v[1]:.3f}")
