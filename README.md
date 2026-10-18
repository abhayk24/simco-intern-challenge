# Cleaner choice challenge: my work

The task is in [CHALLENGE.md](CHALLENGE.md). My step-by-step thinking is in [JOURNAL.md](JOURNAL.md). My first concern, written before any code, is in [FIRST.md](FIRST.md).

## Run it

Python 3.10 or later. No packages to install.

```
python app.py
```

Then open http://localhost:8000. Change a household, price, message or channel and press the button.

- **Live app:** https://cleaner-choice.onrender.com (free plan: the first load after a quiet spell can take about 30-50 seconds)
- **JSON endpoint:** `/api/predict?household_id=new&budget=tight&entrant_price=3.49&channel=online`
  (or POST the same fields as JSON)
- **Saved runs:** `/runs` and `/runs/<id>` in the app, or `python runlog.py list` / `python runlog.py show 7`
- **Run database:** `runs.sqlite` (SQLite), with a JSON copy in `runs_export.json`
- **Tests:** `python -m unittest discover -s tests`

## What is where

| File | What it does |
|---|---|
| `choice_model.py` | The learned choice model (a logit model, standard library only) |
| `build_model.py` | Trains the model and saves it to `model/model_D.json` (run 8) |
| `explain.py` | Answer for one household: chances, likely range, "change one thing" reasons with trust labels |
| `app.py` | The web page and JSON endpoint |
| `runlog.py` | Saves each run: expectation (written first), inputs, outputs, model, data and code version, verdict |
| `experiments/e00_explore.py` | Reproduces the Stage 0 counts and the quick checks quoted in the journal: `python -m experiments.e00_explore` |
| `experiments/` | One file per run (e01 to e11, run 8 is `build_model.py`). Each one states its expectation before it runs |
| `TRUST.md` | Checks, failed and conflicting ideas, what the data can't tell, the real-shopper test |
| `JOURNAL.md` | What I thought and decided at each stage, in order |
| `baseline.py`, `run_baseline.py` | The starter, unchanged |

To redo a run, for example run 3: `python -m experiments.e03_grow_model`. Each rerun is saved as a new run, so the old ones are kept.

| Run | Question |
|---|---|
| 1 | Does the starter beat "always predict the average"? |
| 2 | Are the price, message and channel gaps real? (FIRST.md test) |
| 3 | Which model pieces help on unseen households? |
| 4 | Does a household's past help? |
| 5 | What would each price/message/channel sell, and how sure is the winner? |
| 6 | Can we trust stated reasons? |
| 7 | One-time test on the check set |
| 8 | Save the model for the app |
| 9 | Shuffle test, blank messages, green survey notes |
| 10 | Does the green note explain repeat buying? |
| 11 | How big must the real shopper test be? |

## What the model carries from one visit to the next

Only two things: whether the household bought the new cleaner before, and whether it gave a good or bad note after using it. I tested other history pieces (repeating the last choice, buying nothing right after buying) and they did nothing (run 4).

## Final note

**Recommendation.** Test **online** first, with the **environment** message as the lead. **Price: I do not know yet.** $3.49 sells the most (32% vs 23% at $4.49, run 5). But if a bottle costs the brand more than about $1.16, $4.49 makes more money. That cost isn't in the data.

**Run that changed my view: run 5.** Price was the clearest finding, so I assumed the cheapest price was the answer. Run 5's break-even cost showed that the clearest finding doesn't settle the business choice. Run 9 also made me less sure about online.

**Next test.** One online test with 4 random versions: $3.49 or $4.49, each with environment or cleaning. About 3,700 shoppers if real gaps match the model, about 15,000 if they're half as big (run 11). Measure first purchase and 8-week repeat.

**What would change my choice.** If $4.49 sells within 3 points of $3.49, or $3.49 buyers rarely come back, I pick $4.49. If cleaning beats environment, I switch the message. If the brand shares its cost per bottle, the price question may not need a test at all.

More on trust, failed ideas and the real-shopper test: [TRUST.md](TRUST.md).
