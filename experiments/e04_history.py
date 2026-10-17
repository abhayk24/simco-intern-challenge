"""Run 4: does a household's past help predict its next choice?

Model D = model C + history:
  same_as_last_visit            option = what they chose last time
  none_after_buying_last_visit  "buy nothing" right after a purchase (still have cleaner at home?)
  entrant_bought_before         they bought the entrant at some earlier visit
  entrant_bad_note_before       an earlier entrant use got a bad note (e.g. "It left a stain behind.")
  entrant_good_note_before      an earlier entrant use got a good note

Same 5-fold, by-household cross-validation as run 3. Build set only.
"""

from choice_model import ChoiceModel
from experiments.common import load
from experiments.e03_grow_model import MODELS, Subset, cross_validate
from run_baseline import offers_for
from runlog import run_experiment

EXPECTATION = """
Stage 1 showed households repeat their last choice only 23.1% vs 20.6% by chance, so:
same_as_last_visit small and positive (about +0.15), not clear (p > 0.05).
none_after_buying_last_visit near zero (the starter assumes it is positive; I doubt it).
entrant_bought_before positive; entrant_bad_note_before negative.
Overall D improves CV log loss over C (1.5741) by less than 0.005.
RULE: history is worth carrying between visits only if D beats C in CV log loss.
"""

INPUTS = {"data": "visits.csv (build only)", "folds": 5, "split": "by household", "seed": 11, "ridge": 0.01}

D = MODELS["C"] + ("history",)


def experiment(inputs):
    households, build, _ = load()
    out = {"C": cross_validate(MODELS["C"], build, households, inputs),
           "D": cross_validate(D, build, households, inputs)}
    full = Subset(D, inputs["ridge"]).fit(build, households, offers_for)
    # how often each history feature is "on" for the chosen-or-not option, so we know how much data backs it
    counts = {}
    for ex_feats, _ in full._examples(build, households, offers_for):
        for f in ex_feats:
            for k in f:
                if k in ("same_as_last_visit", "none_after_buying_last_visit", "entrant_bought_before",
                         "entrant_bad_note_before", "entrant_good_note_before"):
                    counts[k] = counts.get(k, 0) + 1
    out["history_weights"] = {k: {"weight": round(full.weights[k], 3), "std_error": round(full.std_errors[k], 3),
                                  "z": round(full.weights[k] / full.std_errors[k], 2), "visits_with_it": counts.get(k, 0)}
                              for k in counts}
    out["rule_history_worth_it"] = out["D"]["cv_log_loss"] < out["C"]["cv_log_loss"]
    return out


if __name__ == "__main__":
    run_id, out = run_experiment("e04 does history help", EXPECTATION, INPUTS, "logit-v2-history", experiment)
    print("C", out["C"]); print("D", out["D"]); print("history worth it:", out["rule_history_worth_it"])
    for k, v in out["history_weights"].items():
        print(f"  {k:30} {v}")
