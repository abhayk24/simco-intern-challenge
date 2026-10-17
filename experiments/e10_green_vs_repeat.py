"""Run 10: does the green survey note explain the "bought the entrant before" effect?

Run 4 found: households that bought the entrant before buy it again more (+0.56).
Two stories fit: (a) trying it made them like it, (b) some households liked it all along.
Run 9 found green-note households buy the entrant much more (+0.82).
If adding the green note shrinks "bought before" a lot, story (b) gains support.

Also: does the green note improve 5-fold cross-validated log loss on build?
Decided now: even if it does, I will NOT add it to the app's model, because the check set
is already used and I cannot test the new model fairly. It goes into the real-shopper test plan instead.
"""

from choice_model import ChoiceModel
from experiments.common import load, score
from experiments.e03_grow_model import folds
from experiments.e09_trust_checks import WithNotes
from run_baseline import offers_for
from runlog import run_experiment

EXPECTATION = """
entrant_bought_before shrinks from +0.56 to about +0.45 but stays clear (z > 2.5).
Green-note weight stays clear (z > 3).
Cross-validated log loss on build improves by about 0.003 over model D (1.5701).
"""

INPUTS = {"data": "visits.csv (build only)", "folds": 5, "seed": 11, "ridge": 0.01}

D = ("brand", "price", "household", "levers", "history")


def cv(cls, visits, households, inputs):
    total = n = 0
    for train, test in folds(visits, inputs["folds"], inputs["seed"]):
        s = score(cls(D, inputs["ridge"]).fit(train, households, offers_for).predict, test, households)
        total += s["log_loss"] * s["visits"]
        n += s["visits"]
    return round(total / n, 4)


def experiment(inputs):
    households, build, _ = load()
    d = ChoiceModel(D, inputs["ridge"]).fit(build, households, offers_for)
    g = WithNotes(D, inputs["ridge"]).fit(build, households, offers_for)
    pick = lambda m, k: {"weight": round(m.weights[k], 3), "z": round(m.weights[k] / m.std_errors[k], 2)}
    return {
        "bought_before_in_D": pick(d, "entrant_bought_before"),
        "bought_before_with_green_note": pick(g, "entrant_bought_before"),
        "green_note_with_history": pick(g, "entrant_if_green_note"),
        "cv_log_loss_D": cv(ChoiceModel, build, households, inputs),
        "cv_log_loss_D_plus_notes": cv(WithNotes, build, households, inputs),
    }


if __name__ == "__main__":
    run_id, out = run_experiment("e10 does the green note explain repeat buying", EXPECTATION, INPUTS,
                                 "logit-v3-notes (not used in app)", experiment)
    for k, v in out.items():
        print(k, v)
