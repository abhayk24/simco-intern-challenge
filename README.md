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
| `experiments/` | One file per run (e01 to e07). Each one states its expectation before it runs |
| `baseline.py`, `run_baseline.py` | The starter, unchanged |

To redo every run: `python -m experiments.e01_baseline` (and so on, up to e07), then `python build_model.py`.

## What the model carries from one visit to the next

Only two things: whether the household bought the new cleaner before, and whether it gave a good or bad note after using it. I tested other history pieces (repeating the last choice, buying nothing right after buying) and they did nothing (run 4).

## Where I stand (not final yet)

- **Channel:** online. Clear in the build data, and it held on new households.
- **Message:** environment is the best bet, but not proven.
- **Price:** I don't know yet. $3.49 sells the most, but if a bottle costs the brand more than about $1.16, $4.49 makes more money. That cost isn't in the data.

The final note comes at the end.
