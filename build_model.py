"""Train model D on all build visits and save it to model/model_D.json for the app.

Also saves 100 refits on resampled households, so the app can show a range
("between X% and Y%") instead of one number that looks more exact than it is.

    python build_model.py
"""

import json
import random
from pathlib import Path

from choice_model import ChoiceModel
from experiments.common import load
from experiments.e05_options_table import resample
from run_baseline import offers_for
from runlog import data_version, run_experiment

GROUPS = ("brand", "price", "household", "levers", "history")
MODEL_PATH = Path(__file__).parent / "model" / "model_D.json"
MODEL_VERSION = "logit-v2-history"

EXPECTATION = """
Same pieces and settings as runs 4, 5 and 7, so the main weights should match run 7's model:
price about -0.45, usual_brand about +0.49, entrant_bought_before about +0.56.
The 100 refits should give price weights between about -0.6 and -0.3.
"""

INPUTS = {"data": "visits.csv (build only)", "groups": list(GROUPS), "ridge": 0.01, "reps": 100, "seed": 21}


def build(inputs):
    households, visits, _ = load()
    main = ChoiceModel(GROUPS, inputs["ridge"]).fit(visits, households, offers_for)
    rng = random.Random(inputs["seed"])
    refits = []
    for _ in range(inputs["reps"]):
        hh, vv = resample(households, visits, rng)
        refits.append(ChoiceModel(GROUPS, inputs["ridge"]).fit(vv, hh, offers_for).weights)
    saved = {"model_version": MODEL_VERSION, "data_version": data_version(), "model": main.to_dict(),
             "refit_weights": refits}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps(saved, indent=1, sort_keys=True), encoding="utf-8")
    prices = sorted(w["price"] for w in refits)
    return {"file": str(MODEL_PATH.name), "weights": {k: round(v, 3) for k, v in main.weights.items()},
            "price_weight_90": [round(prices[5], 3), round(prices[94], 3)]}


if __name__ == "__main__":
    run_id, out = run_experiment("e08 build the saved model for the app", EXPECTATION, INPUTS, MODEL_VERSION, build)
    print(out)
