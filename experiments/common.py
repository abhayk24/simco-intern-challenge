"""Shared helpers for experiments: load data, walk visits in time order, score predictions."""

import math
from collections import Counter, defaultdict

from run_baseline import offers_for, read_rows

OPTIONS = ["entrant", "standard", "budget", "scented", "none"]


def load():
    households = {row["household_id"]: row for row in read_rows("households.csv")}
    return households, read_rows("visits.csv"), read_rows("check_visits.csv")


def by_household(visits):
    groups = defaultdict(list)
    for visit in visits:
        groups[visit["household_id"]].append(visit)
    return groups


def score(predict, visits, households):
    """Walk visits in file order (time order per household), passing only past visits as history.

    predict(household, offers, visit, history) -> {"probabilities": {...}}
    """
    history = defaultdict(list)
    n = correct = 0
    log_loss = 0.0
    expected = Counter()
    observed = Counter()
    for visit in visits:
        hid = visit["household_id"]
        probs = predict(households[hid], offers_for(visit), visit, history[hid])["probabilities"]
        actual = visit["choice"]
        n += 1
        correct += max(probs, key=probs.get) == actual
        log_loss -= math.log(max(probs.get(actual, 0.0), 1e-12))
        for option in OPTIONS:
            expected[option] += probs.get(option, 0.0)
        observed[actual] += 1
        history[hid].append(visit)
    return {
        "visits": n,
        "top_choice_matched": round(correct / n, 4),
        "log_loss": round(log_loss / n, 4),
        "share_observed": {o: round(observed[o] / n, 4) for o in OPTIONS},
        "share_expected": {o: round(expected[o] / n, 4) for o in OPTIONS},
    }
