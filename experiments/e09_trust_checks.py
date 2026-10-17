"""Run 9: three more checks on the pieces the recommendation leans on. Build set only.

1. Shuffle test for message. Shuffle which message each visit saw (within the same channel),
   refit, and see how often a FAKE message gap is as big as the real one.
2. Online without blank messages. Blank messages sit mostly in retail (stage 0). Drop those
   visits and check whether "online is better" survives.
3. A household idea nobody tested yet. Do households whose survey note sounds green
   ("go down the drain", "fewer plastic bottles") buy the entrant more or respond more to the
   environment message? Do households that "do not trust green claims" respond less?
"""

import random

from choice_model import ChoiceModel
from experiments.common import load
from experiments.e03_grow_model import wald
from run_baseline import offers_for
from runlog import run_experiment

EXPECTATION = """
1. Shuffle test p-value for message about 0.05-0.10, close to the Wald p of 0.057. Still not proven.
2. Without blank-message visits, the online weight stays about +0.3 with z about 2.
   So the blank-message pattern is not what makes online look better.
3. Green households: entrant weight about +0.3 but not clear (|z| < 2). Environment message
   extra effect for green households: not clear. For "do not trust green claims": negative, not clear.
   Too few households to settle any of it. This stays an open idea.
"""

INPUTS = {"data": "visits.csv (build only)", "model": "C = brand, price, household, levers",
          "shuffles": 200, "seed": 9, "ridge": 0.01}

C = ("brand", "price", "household", "levers")
MSG = ["entrant_msg_cleaning", "entrant_msg_environment", "entrant_msg_low_odor"]
GREEN = ("I worry about cleaners that go down the drain.", "I want to use fewer plastic bottles.")
SKEPTIC = "I do not trust green claims without proof."


class WithNotes(ChoiceModel):
    """Model C plus three household-note pieces for the entrant."""

    def _rows(self, offers, household, visit, history):
        rows = super()._rows(offers, household, visit, history)
        note = household.get("note", "")
        green = any(g in note for g in GREEN)
        skeptic = SKEPTIC in note
        for name, f in rows:
            if name != "entrant":
                continue
            env = "entrant_msg_environment" in f
            if green:
                f["entrant_if_green_note"] = 1.0
                if env:
                    f["environment_msg_if_green_note"] = 1.0
            if skeptic and env:
                f["environment_msg_if_skeptic_note"] = 1.0
        return rows


def shuffled(visits, rng):
    out = [dict(v) for v in visits]
    for channel in ("online", "retail"):
        idx = [i for i, v in enumerate(out) if v["channel"] == channel and v["entrant_message"]]
        msgs = [out[i]["entrant_message"] for i in idx]
        rng.shuffle(msgs)
        for i, m in zip(idx, msgs):
            out[i]["entrant_message"] = m
    return out


def piece(model, name):
    w, se = model.weights[name], model.std_errors[name]
    return {"weight": round(w, 3), "std_error": round(se, 3), "z": round(w / se, 2)}


def experiment(inputs):
    households, build, _ = load()
    rng = random.Random(inputs["seed"])

    real = ChoiceModel(C, inputs["ridge"]).fit(build, households, offers_for)
    real_stat = wald(real, MSG)["stat"]
    fake = [wald(ChoiceModel(C, inputs["ridge"]).fit(shuffled(build, rng), households, offers_for), MSG)["stat"]
            for _ in range(inputs["shuffles"])]
    check1 = {"real_wald_stat": real_stat, "share_of_shuffles_as_big": round(sum(s >= real_stat for s in fake) / len(fake), 3),
              "shuffle_stat_90th": round(sorted(fake)[int(0.9 * len(fake))], 2)}

    no_blank = [v for v in build if v["entrant_message"]]
    m2 = ChoiceModel(C, inputs["ridge"]).fit(no_blank, households, offers_for)
    check2 = {"visits_kept": len(no_blank), "online_all_visits": piece(real, "entrant_online"),
              "online_without_blank_messages": piece(m2, "entrant_online")}

    m3 = WithNotes(C, inputs["ridge"]).fit(build, households, offers_for)
    build_ids = {v["household_id"] for v in build}
    count = lambda test: sum(1 for h in build_ids if test(households[h].get("note", "")))
    check3 = {"households_green": count(lambda n: any(g in n for g in GREEN)),
              "households_skeptic": count(lambda n: SKEPTIC in n),
              "pieces": {k: piece(m3, k) for k in ("entrant_if_green_note", "environment_msg_if_green_note",
                                                   "environment_msg_if_skeptic_note") if k in m3.weights}}
    return {"1_message_shuffle_test": check1, "2_online_without_blank": check2, "3_green_note_idea": check3}


if __name__ == "__main__":
    run_id, out = run_experiment("e09 trust checks: shuffle test, blank messages, green notes", EXPECTATION,
                                 INPUTS, "logit-v1", experiment)
    for k, v in out.items():
        print(k, v)
