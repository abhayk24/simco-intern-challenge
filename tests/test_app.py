import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from app import Handler
from explain import answer


class AnswerTests(unittest.TestCase):
    def test_changed_price_gives_new_result(self):
        cheap = answer({"household_id": "new", "entrant_price": "3.49"})["probabilities"]["entrant"]
        dear = answer({"household_id": "new", "entrant_price": "5.49"})["probabilities"]["entrant"]
        self.assertGreater(cheap, dear)

    def test_existing_household_uses_its_history(self):
        with_h = answer({"household_id": "H001"})
        without = answer({"household_id": "H001", "use_history": "no"})
        self.assertEqual(len(with_h["past_visits"]), 6)
        self.assertEqual(without["past_visits"], [])

    def test_every_reason_has_trust_and_evidence(self):
        r = answer({"household_id": "new", "usual_brand": "standard", "bought_entrant_before": "yes",
                    "entrant_note": "bad", "entrant_price": "3.49"})
        self.assertTrue(r["reasons"])
        for reason in r["reasons"]:
            self.assertIn(reason["trust"], ("solid", "weak", "not proven"))
            self.assertTrue(reason["evidence"])

    def test_range_contains_the_main_answer(self):
        r = answer({"household_id": "H120", "channel": "retail"})
        low, high = r["entrant_likely_range"]
        self.assertLessEqual(low, r["probabilities"]["entrant"])
        self.assertGreaterEqual(high, r["probabilities"]["entrant"])

    def test_price_outside_data_gives_warning(self):
        r = answer({"household_id": "new", "entrant_price": "6.99"})
        self.assertTrue(any("outside" in w for w in r["warnings"]))

    def test_bad_inputs_are_refused(self):
        for bad in ({"household_id": "H999"}, {"entrant_price": "abc"}, {"channel": "tv"}, {"entrant_price": "-1"}):
            with self.assertRaises(ValueError):
                answer(bad)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_get_and_post_predict(self):
        got = json.load(urlopen(f"{self.base}/api/predict?household_id=new&entrant_price=3.49"))
        posted = json.load(urlopen(Request(f"{self.base}/api/predict", method="POST",
                                           data=json.dumps({"household_id": "new", "entrant_price": 3.49}).encode())))
        self.assertEqual(got["probabilities"], posted["probabilities"])

    def test_pages_load(self):
        for path in ("/", "/?household_id=H001", "/runs", "/runs/1", "/health"):
            self.assertEqual(urlopen(self.base + path).status, 200)

    def test_bad_input_is_400(self):
        with self.assertRaises(HTTPError) as e:
            urlopen(f"{self.base}/api/predict?household_id=H999")
        self.assertEqual(e.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
