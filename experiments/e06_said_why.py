"""Run 6: can we trust what households say about why they chose?

For each said_why answer, I check it against facts we already have:
- "I bought the cleaner I often buy."  -> did they pick their usual brand from the survey?
- "I know this brand/cleaner."         -> contradiction if it is their FIRST entrant purchase
- "I wanted to try a new cleaner."     -> should be a first entrant purchase
- "did not need / still had / left"    -> should be "none"
- "still had cleaner at home"          -> did they actually buy something last visit?
- "did not see a reason to pay more"   -> did they pick the cheapest product on the shelf?
- "less waste" / "refillable bottle"   -> only the entrant claims this
- "fresh scent"                        -> only scented claims this
- "tough stains"                       -> only standard claims this
- "less smell"                         -> entrant (low odor) or budget (no added scent)

Also: who answers at all, and do entrant buyers' reasons follow the shelf message?
Build set only.
"""

from collections import Counter, defaultdict

from experiments.common import by_household, load
from runlog import run_experiment

EXPECTATION = """
"I bought the cleaner I often buy" matches the survey usual brand in at least 70% of cases.
"I know this brand/cleaner" shows up on some first-ever entrant purchases (over 5%): a clear contradiction.
"none" reasons said alongside a purchase: under 5%.
"less waste"/"refillable" come from entrant buyers over 90% of the time, and entrant buyers give
them about twice as often when the shelf said "environment".
People who bought nothing answer less often than buyers.
Overall: most answers fit the facts, but 10-25% of checkable answers contradict them.
So said_why is a weak hint, not the truth, and I will not use it as a model input.
"""

INPUTS = {"data": "visits.csv (build only)", "field": "said_why"}


def cheapest(v):
    prices = {p: float(v[f"{p}_price"]) for p in ("entrant", "standard", "budget", "scented")}
    return min(prices, key=prices.get)


def check(reason, v, household, past):
    """True = fits the facts, False = contradicts them, None = cannot check."""
    choice = v["choice"]
    first_entrant = choice == "entrant" and not any(h["choice"] == "entrant" for h in past)
    if reason == "I bought the cleaner I often buy.":
        return None if not household["usual_brand"] else choice == household["usual_brand"]
    if reason in ("I know this brand.", "I know this cleaner."):
        return False if first_entrant else (None if choice != "none" else False)
    if reason == "I wanted to try a new cleaner.":
        return first_entrant
    if reason in ("I did not need cleaner today.", "I left without buying one."):
        return choice == "none"
    if reason == "I still had cleaner at home.":
        return choice == "none" and bool(past) and past[-1]["choice"] != "none"
    if reason == "I did not see a reason to pay more.":
        return choice == cheapest(v)
    if reason in ("I thought it would make less waste.", "I wanted a refillable bottle."):
        return choice == "entrant"
    if reason == "I wanted the fresh scent.":
        return choice == "scented"
    if reason == "I expected it to clean tough stains.":
        return choice == "standard"
    if reason == "I wanted less smell.":
        return choice in ("entrant", "budget")
    return None  # "clean well", "price fit my budget", "price seemed fair": too vague to check


def experiment(inputs):
    households, build, _ = load()
    per_reason = defaultdict(Counter)
    answered = Counter()
    seen = Counter()
    waste_by_message = defaultdict(Counter)
    for visits in by_household(build).values():
        for i, v in enumerate(visits):
            seen[v["choice"]] += 1
            reason = v["said_why"]
            if v["choice"] == "entrant":
                key = "waste_or_refill" if reason in ("I thought it would make less waste.",
                                                      "I wanted a refillable bottle.") else ("other" if reason else "blank")
                waste_by_message[v["entrant_message"] or "(blank)"][key] += 1
            if not reason:
                continue
            answered[v["choice"]] += 1
            result = check(reason, v, households[v["household_id"]], visits[:i])
            per_reason[reason][{True: "fits", False: "contradicts", None: "cannot_check"}[result]] += 1
            per_reason[reason][f"chose_{v['choice']}"] += 1

    fits = sum(c["fits"] for c in per_reason.values())
    contra = sum(c["contradicts"] for c in per_reason.values())
    return {
        "per_reason": {r: dict(c) for r, c in sorted(per_reason.items(), key=lambda x: -sum(x[1][k] for k in ("fits", "contradicts", "cannot_check")))},
        "checkable_answers": fits + contra,
        "share_contradicting": round(contra / (fits + contra), 3),
        "answer_rate_by_choice": {c: round(answered[c] / seen[c], 3) for c in seen},
        "entrant_buyers_waste_reason_by_message": {
            m: {"n_answered": c["waste_or_refill"] + c["other"],
                "share_waste_or_refill": round(c["waste_or_refill"] / max(1, c["waste_or_refill"] + c["other"]), 3)}
            for m, c in sorted(waste_by_message.items())},
    }


if __name__ == "__main__":
    run_id, out = run_experiment("e06 can we trust said_why", EXPECTATION, INPUTS, "check-v1", experiment)
    for r, c in out["per_reason"].items():
        print(f"{r:40} {c}")
    for k in ("checkable_answers", "share_contradicting", "answer_rate_by_choice", "entrant_buyers_waste_reason_by_message"):
        print(k, out[k])
