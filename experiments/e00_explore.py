"""Stage 0 exploration, plus the quick checks quoted in the journal.

These counts were first run as one-off snippets while exploring. This file was added
later (after Stage 5) so anyone can reproduce every number quoted from them.
It only reads the data and prints; it changes nothing and is not a saved run.

    python -m experiments.e00_explore
"""

from collections import Counter, defaultdict

from experiments.common import by_household, load


def share_table(title, visits, key):
    """New-cleaner purchase rate (and 'none' rate) for each value of key(visit)."""
    groups = defaultdict(list)
    for v in visits:
        groups[key(v)].append(v["choice"])
    print(f"\n{title}")
    for k in sorted(groups):
        c, n = Counter(groups[k]), len(groups[k])
        print(f"  {str(k):28} n={n:4}  entrant={c['entrant'] / n:5.1%}  none={c['none'] / n:5.1%}")


def cross_table(title, visits, a, b):
    """How often each pair of values appears together. Even counts = levers mixed fairly."""
    t = Counter((a(v), b(v)) for v in visits)
    rows, cols = sorted({k[0] for k in t}), sorted({k[1] for k in t})
    print(f"\n{title}")
    print(" " * 14 + "".join(str(c or "(blank)").ljust(13) for c in cols))
    for r in rows:
        print(str(r or "(blank)").ljust(14) + "".join(str(t[(r, c)]).ljust(13) for c in cols))


def main():
    households, build, check = load()
    hh = lambda v, k: households[v["household_id"]][k]

    print("== 1. What people bought ==")
    print("  build:", Counter(v["choice"] for v in build).most_common())
    print("  check:", Counter(v["choice"] for v in check).most_common())

    print("\n== 2. Were the levers mixed fairly? (build) ==")
    cross_table("price x message", build, lambda v: v["entrant_price"], lambda v: v["entrant_message"])
    cross_table("message x channel", build, lambda v: v["entrant_message"], lambda v: v["channel"])
    cross_table("budget x price", build, lambda v: hh(v, "budget"), lambda v: v["entrant_price"])
    one_price = sum(len({v["entrant_price"] for v in g}) == 1 for g in by_household(build).values())
    print(f"\n  households that saw only one entrant price: {one_price}")

    print("\n== 3. Raw new-cleaner purchase rate (build) ==")
    share_table("by price", build, lambda v: v["entrant_price"])
    share_table("by message", build, lambda v: v["entrant_message"] or "(blank)")
    share_table("by channel", build, lambda v: v["channel"])

    print("\n== 4. Free text (build) ==")
    print("  after-use notes for the entrant:",
          Counter(v["after_use_note"] for v in build if v["choice"] == "entrant" and v["after_use_note"]).most_common())

    print("\n== 5. Quick checks quoted later ==")
    groups = by_household(build)
    counts = [sum(v["choice"] == "entrant" for v in g) for g in groups.values()]
    p = sum(counts) / len(build)
    mean = sum(counts) / len(counts)
    var = sum((c - mean) ** 2 for c in counts) / (len(counts) - 1)
    print(f"  spread of entrant buys per household: {var:.2f} (vs {6 * p * (1 - p):.2f} if all households were alike)")

    same = sum(a["choice"] == b["choice"] for g in groups.values() for a, b in zip(g, g[1:]))
    pairs = sum(len(g) - 1 for g in groups.values())
    chance = sum((c / len(build)) ** 2 for c in Counter(v["choice"] for v in build).values())
    print(f"  repeats last choice: {same / pairs:.1%} (vs {chance:.1%} if visits were independent)")

    before, never = [0, 0], [0, 0]
    for g in groups.values():
        seen = False
        for v in g:
            bucket = before if seen else never
            bucket[0] += v["choice"] == "entrant"
            bucket[1] += 1
            seen = seen or v["choice"] == "entrant"
    print(f"  buys entrant when bought it before: {before[0] / before[1]:.1%} (n={before[1]}); "
          f"never before: {never[0] / never[1]:.1%} (n={never[1]})")

    for name, rows in (("build", build), ("check", check)):
        by_reason = defaultdict(Counter)
        for v in rows:
            if v["said_why"]:
                by_reason[v["said_why"]][v["choice"]] += 1
        single = sum(len(c) == 1 for c in by_reason.values())
        top = sum(c.most_common(1)[0][1] for c in by_reason.values())
        total = sum(sum(c.values()) for c in by_reason.values())
        print(f"  said_why ({name}): {single}/{len(by_reason)} reasons always go with one choice; "
              f"the reason gives the choice {top / total:.1%} of the time")


if __name__ == "__main__":
    main()
