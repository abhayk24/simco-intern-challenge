"""Run 5: what would each test option sell? And how sure are we which one is best?

For every price x message x channel (3 x 3 x 2 = 18 options) I ask model D:
"if a household walks in for the first time and sees this, what is the chance it buys the entrant?"
Competitors are at their usual prices. Nobody has bought the entrant before (it is a launch).
I average over the build households.

To see how sure we are, I repeat everything 200 times on resampled households (refitting the
model each time) and count how often each option comes out on top.

Profit needs the brand's cost per bottle, which we do not have. So I work out the
break-even cost: the cost per bottle above which the higher price makes more money.
"""

import random
from itertools import product

from choice_model import ChoiceModel, MESSAGES
from experiments.common import by_household, load
from run_baseline import offers_for
from runlog import run_experiment

EXPECTATION = """
$3.49 gives the highest entrant share in over 99% of resamples.
environment is the best message in about 70% of resamples; low_odor best in under 5%.
online beats retail in about 98% of resamples.
Best option (3.49, environment, online) has entrant share about 30-35%.
Revenue per shopper is also highest at $3.49.
Break-even cost per bottle for $4.49 over $3.49 is about $1.50, with a wide interval.
So the best PRICE for profit depends on a cost we do not know.
"""

INPUTS = {"data": "visits.csv (build only)", "model": "D = brand, price, household, levers, history",
          "reps": 200, "seed": 5, "ridge": 0.01,
          "competitor_prices": {"standard": 3.99, "budget": 2.99, "scented": 4.29},
          "prices": [3.49, 4.49, 5.49], "messages": MESSAGES, "channels": ["online", "retail"]}

GROUPS = ("brand", "price", "household", "levers", "history")


def shares(model, households, inputs):
    """Average first-visit chance of buying the entrant, for each option."""
    comp = inputs["competitor_prices"]
    table = {}
    for price, message, channel in product(inputs["prices"], inputs["messages"], inputs["channels"]):
        offers = [{"product_id": "entrant", "price": price, "message": message}] + \
                 [{"product_id": p, "price": c} for p, c in comp.items()]
        visit = {"channel": channel}
        total = sum(model.predict(h, offers, visit, [])["probabilities"]["entrant"] for h in households)
        table[(price, message, channel)] = total / len(households)
    return table


def price_level(table, price):
    rows = [v for (p, _, _), v in table.items() if p == price]
    return sum(rows) / len(rows)


def break_even(table, low, high):
    """Cost c where (low - c) * share_low == (high - c) * share_high."""
    s_low, s_high = price_level(table, low), price_level(table, high)
    return (low * s_low - high * s_high) / (s_low - s_high)


def resample(households, build, rng):
    groups = list(by_household(build).values())
    new_households, new_visits = {}, []
    for i, g in enumerate(rng.choices(groups, k=len(groups))):
        hid = f"R{i}"  # a fresh id so two copies of one household do not share a history
        new_households[hid] = {**households[g[0]["household_id"]], "household_id": hid}
        new_visits += [{**v, "household_id": hid} for v in g]
    return new_households, new_visits


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def experiment(inputs):
    households, build, _ = load()
    rng = random.Random(inputs["seed"])
    model = ChoiceModel(GROUPS, inputs["ridge"]).fit(build, households, offers_for)
    point = shares(model, list(households.values())[:200], inputs)  # build households only

    reps = []
    for _ in range(inputs["reps"]):
        hh, visits = resample(households, build, rng)
        m = ChoiceModel(GROUPS, inputs["ridge"]).fit(visits, hh, offers_for)
        reps.append((m, shares(m, list(hh.values()), inputs)))

    def best_share(f):
        counts = {}
        for _, t in reps:
            k = f(max(t, key=t.get))
            counts[k] = counts.get(k, 0) + 1
        return {str(k): round(v / len(reps), 3) for k, v in sorted(counts.items(), key=lambda x: -x[1])}

    message_best = {}
    for m, _ in reps:
        top = max(MESSAGES, key=lambda x: m.weights.get(f"entrant_msg_{x}", 0.0))
        message_best[top] = message_best.get(top, 0) + 1

    out = {"options": {}}
    for key, value in sorted(point.items(), key=lambda x: -x[1]):
        values = [t[key] for _, t in reps]
        out["options"][" / ".join(map(str, key))] = {
            "entrant_share": round(value, 3),
            "share_90": [round(pct(values, 0.05), 3), round(pct(values, 0.95), 3)],
            "revenue_per_shopper": round(key[0] * value, 3),
        }
    out["chance_best_option_by_share"] = best_share(lambda k: k)
    out["chance_best_price_by_share"] = best_share(lambda k: k[0])
    out["chance_best_message"] = {k: round(v / len(reps), 3) for k, v in message_best.items()}
    out["chance_online_beats_retail"] = round(sum(m.weights["entrant_online"] > 0 for m, _ in reps) / len(reps), 3)
    out["price_level_share"] = {p: round(price_level(point, p), 3) for p in inputs["prices"]}
    for low, high in [(3.49, 4.49), (4.49, 5.49)]:
        values = [break_even(t, low, high) for _, t in reps]
        out[f"break_even_cost_{low}_vs_{high}"] = {"point": round(break_even(point, low, high), 2),
                                                   "interval_90": [round(pct(values, 0.05), 2), round(pct(values, 0.95), 2)]}
    return out


if __name__ == "__main__":
    run_id, out = run_experiment("e05 options table with uncertainty", EXPECTATION, INPUTS, "logit-v2-history", experiment)
    print(f"{'option':30} {'share':>6}  {'90% range':>16}  revenue/shopper")
    for k, v in out["options"].items():
        print(f"{k:30} {v['entrant_share']:>6}  {str(v['share_90']):>16}  {v['revenue_per_shopper']}")
    for k, v in out.items():
        if k != "options":
            print(k, v)
