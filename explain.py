"""Turn the saved model into an answer for one household, with reasons and how far to trust them.

The reasons are "change one thing" reasons: for each piece of the model that applies to this
household, I switch just that piece off and see how much the entrant's chance moves.
Each reason gets a trust label from the data (how sure the weight is) plus a note
pointing to the saved run that backs it or doubts it.
"""

import json
from functools import lru_cache
from pathlib import Path

from choice_model import ChoiceModel, softmax
from experiments.common import load

MODEL_PATH = Path(__file__).parent / "model" / "model_D.json"
PRODUCTS = ["entrant", "standard", "budget", "scented"]
USUAL_PRICES = {"entrant": 4.49, "standard": 3.99, "budget": 2.99, "scented": 4.29}
TESTED = {"entrant": (3.49, 5.49), "standard": (3.49, 3.99), "budget": (2.99, 2.99), "scented": (3.79, 4.29)}

# Plain words for each model piece, and what the saved runs say about it.
PIECES = {
    "usual_brand": ("Their usual brand (from the survey) is {usual}", None,
                    "Run 3: +0.49, clear. But run 6: the survey brand disagrees with what people say 9 of 31 times."),
    "price_if_tight_budget": ("Tight budget makes price hurt more", None, "Run 3: small and not clear (-0.08 +/- 0.04)."),
    "price_if_flexible_budget": ("Flexible budget makes price hurt less", None, "Run 3: about zero."),
    "entrant_msg_cleaning": ("Shelf message: cleaning", "not proven",
                             "Run 3: messages failed my strict rule (p=0.057). Run 5: cleaning is best in 17.5% of refits."),
    "entrant_msg_environment": ("Shelf message: environment", "not proven",
                                "Run 3: messages failed my strict rule (p=0.057). Run 5: environment is best in 81.5% of refits."),
    "entrant_msg_low_odor": ("Shelf message: low odor", "not proven",
                             "Run 3: messages failed my strict rule (p=0.057). Run 5: low odor is best in 1% of refits."),
    "entrant_online": ("Sold online", None,
                       "Run 3: p=0.026, failed my strict p<0.01 rule. Run 7: held on new households (22% vs 14%)."),
    "entrant_bought_before": ("They bought the new cleaner before", None,
                              "Run 4: clear (z=3.4). But it may be that they liked it all along, not that trying it made them like it."),
    "entrant_bad_note_before": ("They gave it a bad note after use", None, "Run 4: a hint only, based on 51 visits."),
    "entrant_good_note_before": ("They gave it a good note after use", None, "Run 4: not clear, based on 76 visits."),
    "same_as_last_visit": ("Same product as their last visit", None, "Run 4: zero effect."),
    "none_after_buying_last_visit": ("They bought something last visit", None, "Run 4: about zero; the starter's idea is not supported."),
}

MODEL_CAVEATS = [
    "On 50 new households the model beat the starter only slightly (log loss 1.552 vs 1.563), and that win is not certain (run 7).",
    "It slightly oversells the new cleaner: 21.0% predicted vs 18.3% real on new households (run 7).",
    "Most of any single choice cannot be predicted from these fields (runs 3 and 7).",
    "Income, cleaning spend, bathrooms and the survey note are NOT used. I did not test them.",
    "Stated reasons (said_why) are shown but not used: they mostly repeat each product's own claims (run 6).",
]


@lru_cache(maxsize=1)
def saved():
    data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    main = ChoiceModel.from_dict(data["model"])
    refits = [ChoiceModel.from_dict({**data["model"], "weights": w, "std_errors": {}}) for w in data["refit_weights"]]
    return data, main, refits


@lru_cache(maxsize=1)
def dataset():
    households, build, check = load()
    visits = {}
    for v in build + check:
        visits.setdefault(v["household_id"], []).append(v)
    return households, visits


def trust_label(name, model):
    fixed = PIECES.get(name, (None, None))[1]
    if fixed:
        return fixed
    z = abs(model.weights[name] / model.std_errors[name]) if model.std_errors.get(name) else 0
    return "solid" if z >= 3 else "weak" if z >= 2 else "not proven"


def entrant_chance(model, household, offers, visit, history, drop=None):
    parts = model.score_parts(household, offers, visit, history)
    names = list(parts)
    scores = [sum(v for k, v in parts[n].items() if k != drop) for n in names]
    return dict(zip(names, softmax(scores)))["entrant"]


def build_inputs(params):
    """Read the request into (household, offers, visit, history, warnings). Raises ValueError on bad input."""
    warnings = []
    households, visits = dataset()
    hid = params.get("household_id", "new").strip() or "new"

    if hid != "new":
        if hid not in households:
            raise ValueError(f"Unknown household_id {hid!r}. Use H001-H250 or 'new'.")
        household = households[hid]
        history = visits.get(hid, []) if params.get("use_history", "yes") == "yes" else []
        if history:
            warnings.append(f"Using this household's {len(history)} recorded visits as history (predicting its next visit).")
    else:
        budget = params.get("budget", "middle")
        if budget not in ("tight", "middle", "flexible"):
            raise ValueError("budget must be tight, middle or flexible.")
        usual = params.get("usual_brand", "")
        if usual not in ("", "standard", "budget", "scented"):
            raise ValueError("usual_brand must be standard, budget, scented or blank.")
        household = {"household_id": "new", "budget": budget, "usual_brand": usual}
        history = []
        if params.get("bought_entrant_before") == "yes":
            note = {"good": "It removed the stain I noticed.", "bad": "It left a stain behind."}.get(params.get("entrant_note", ""), "")
            history.append({"choice": "entrant", "after_use_note": note})
        last = params.get("last_choice", "")
        if last and last not in PRODUCTS + ["none"]:
            raise ValueError("last_choice must be a product, none, or blank.")
        if last and not (history and last == "entrant"):
            history.append({"choice": last, "after_use_note": ""})

    message = params.get("message", "environment")
    if message not in ("cleaning", "environment", "low_odor", ""):
        raise ValueError("message must be cleaning, environment, low_odor or blank.")
    channel = params.get("channel", "online")
    if channel not in ("online", "retail"):
        raise ValueError("channel must be online or retail.")

    offers = []
    for p in PRODUCTS:
        try:
            price = float(params.get(f"{p}_price", USUAL_PRICES[p]))
        except ValueError:
            raise ValueError(f"{p}_price must be a number.")
        if price <= 0 or price > 50:
            raise ValueError(f"{p}_price must be between 0 and 50.")
        lo, hi = TESTED[p]
        if not lo <= price <= hi:
            warnings.append(f"{p} price ${price:.2f} is outside the prices in the data (${lo:.2f}-${hi:.2f}). This is a guess past the data.")
        offer = {"product_id": p, "price": price}
        if p == "entrant":
            offer["message"] = message
        offers.append(offer)
    return household, offers, {"channel": channel}, history, warnings


def answer(params):
    data, model, refits = saved()
    household, offers, visit, history, warnings = build_inputs(params)
    probs = model.predict(household, offers, visit, history)["probabilities"]
    p_entrant = probs["entrant"]

    spread = sorted(entrant_chance(m, household, offers, visit, history) for m in refits)
    low, high = spread[int(0.05 * len(spread))], spread[int(0.95 * len(spread)) - 1]

    # "change one thing" reasons
    parts = model.score_parts(household, offers, visit, history)
    present = sorted({k for f in parts.values() for k in f if k in PIECES})
    reasons = []
    for name in present:
        change = p_entrant - entrant_chance(model, household, offers, visit, history, drop=name)
        if abs(change) < 0.002:
            continue
        text = PIECES[name][0].format(usual=household.get("usual_brand") or "?")
        reasons.append({"piece": name, "says": text, "moves_entrant_chance_by_points": round(100 * change, 1),
                        "trust": trust_label(name, model), "evidence": PIECES[name][2]})
    entrant_price = offers[0]["price"]
    if abs(entrant_price - USUAL_PRICES["entrant"]) > 1e-9:
        at_usual = [dict(o, price=USUAL_PRICES["entrant"]) if o["product_id"] == "entrant" else o for o in offers]
        change = p_entrant - model.predict(household, at_usual, visit, history)["probabilities"]["entrant"]
        reasons.append({"piece": "price", "says": f"Price ${entrant_price:.2f} instead of the usual $4.49",
                        "moves_entrant_chance_by_points": round(100 * change, 1), "trust": trust_label("price", model),
                        "evidence": "Runs 2, 3 and 5: clear. Run 7: held on new households (25.5% at $3.49 vs 7.7% at $5.49)."})
    reasons.sort(key=lambda r: -abs(r["moves_entrant_chance_by_points"]))

    # what if the entrant price were different (everything else the same)
    what_if = {}
    for price in (3.49, 4.49, 5.49):
        changed = [dict(o, price=price) if o["product_id"] == "entrant" else o for o in offers]
        what_if[f"{price:.2f}"] = round(model.predict(household, changed, visit, history)["probabilities"]["entrant"], 3)

    top = max(probs, key=probs.get)
    headline = (f"New cleaner: {100 * p_entrant:.0f}% chance (likely range {100 * low:.0f}-{100 * high:.0f}%). "
                f"Most likely choice: {top} ({100 * probs[top]:.0f}%).")
    if reasons:
        r = reasons[0]
        headline += f" Biggest reason: {r['says'].lower()} ({r['moves_entrant_chance_by_points']:+.1f} points, {r['trust']})."

    past = [{"choice": h.get("choice"), "said_why": h.get("said_why", ""), "after_use_note": h.get("after_use_note", "")}
            for h in history]
    return {
        "household": {k: household.get(k, "") for k in ("household_id", "budget", "usual_brand", "income_band", "note")},
        "offers": offers, "channel": visit["channel"],
        "probabilities": {k: round(v, 3) for k, v in sorted(probs.items(), key=lambda x: -x[1])},
        "entrant_likely_range": [round(low, 3), round(high, 3)],
        "headline": headline, "reasons": reasons, "what_if_entrant_price": what_if,
        "past_visits": past, "warnings": warnings, "model_caveats": MODEL_CAVEATS,
        "model_version": data["model_version"], "data_version": data["data_version"], "model_run": 8,
    }
