"""Run 1: record the starter model, and a 'know nothing' reference, before changing anything.

Reference model: always predict the overall build-set shares. Any real model must beat it.
"""

from collections import Counter

from baseline import predict as starter_predict
from experiments.common import OPTIONS, load, score
from runlog import run_experiment

EXPECTATION = """
Starter: top choice matched about 29% (build) and 28% (check), as run_baseline.py prints.
I expect its log loss to be WORSE than the constant-share reference, because its weights are
hand-set and it underpredicts the entrant (15.9% vs 21.1%). Constant reference log loss should
be near the entropy of the shares, about 1.59 on build.
"""


def experiment(inputs):
    households, build, check = load()
    counts = Counter(v["choice"] for v in build)
    shares = {o: counts[o] / len(build) for o in OPTIONS}
    constant = lambda *args: {"probabilities": shares}
    return {
        "starter_build": score(starter_predict, build, households),
        "starter_check": score(starter_predict, check, households),
        "constant_build_in_sample": score(constant, build, households),
        "constant_check": score(constant, check, households),
        "note": "The starter is not trained on any data, so scoring it on check uses up nothing.",
    }


if __name__ == "__main__":
    run_id, out = run_experiment(
        name="e01 starter model and constant-share reference",
        expectation=EXPECTATION,
        inputs={"models": ["baseline.py as given", "constant build shares"]},
        model_version="starter-v0",
        fn=experiment,
    )
    for key, value in out.items():
        print(key, value)
