import unittest

from choice_model import ChoiceModel
from experiments.common import load
from run_baseline import offers_for

HOUSEHOLDS, BUILD, _ = load()
MODEL = ChoiceModel(groups=("brand", "price", "household", "levers", "history")).fit(BUILD, HOUSEHOLDS, offers_for)
VISIT = BUILD[2]
HOUSEHOLD = HOUSEHOLDS[VISIT["household_id"]]
PAST = [v for v in BUILD if v["household_id"] == VISIT["household_id"]][:2]


class ChoiceModelTests(unittest.TestCase):
    def test_probabilities_cover_all_options_and_sum_to_one(self):
        probs = MODEL.predict(HOUSEHOLD, offers_for(VISIT), VISIT, PAST)["probabilities"]
        self.assertEqual(set(probs), {"entrant", "standard", "budget", "scented", "none"})
        self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_higher_entrant_price_lowers_entrant_chance(self):
        offers = offers_for(VISIT)
        dearer = [dict(o, price=o["price"] + 1) if o["product_id"] == "entrant" else o for o in offers]
        low = MODEL.predict(HOUSEHOLD, offers, VISIT, PAST)["probabilities"]["entrant"]
        high = MODEL.predict(HOUSEHOLD, dearer, VISIT, PAST)["probabilities"]["entrant"]
        self.assertLess(high, low)

    def test_future_visits_do_not_change_training_examples_for_earlier_visits(self):
        # Features for visit k must only use visits before k. Changing a LATER choice must not
        # change the features built for an earlier visit.
        rows = [dict(v) for v in BUILD if v["household_id"] == "H001"]
        before = MODEL._examples(rows, HOUSEHOLDS, offers_for)
        rows[-1]["choice"] = "entrant" if rows[-1]["choice"] != "entrant" else "none"
        after = MODEL._examples(rows, HOUSEHOLDS, offers_for)
        self.assertEqual([f for f, _ in before[:-1]], [f for f, _ in after[:-1]])

    def test_reason_names_the_top_choice(self):
        result = MODEL.predict(HOUSEHOLD, offers_for(VISIT), VISIT, PAST)
        top = max(result["probabilities"], key=result["probabilities"].get)
        self.assertIn(top, result["reason"])

    def test_missing_usual_brand_and_message_still_predict(self):
        household = {**HOUSEHOLD, "usual_brand": ""}
        offers = [dict(o, message="") for o in offers_for(VISIT)]
        probs = MODEL.predict(household, offers, VISIT, [])["probabilities"]
        self.assertAlmostEqual(sum(probs.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
