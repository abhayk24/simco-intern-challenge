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

## Final note( it has 200 word limit, but I added detail final note thats break assignment rule but it would be better to understand , what I understood from assignment)

**Recommendation.** I recommend testing the product online first, with the environment message as one of the options. My model predicts more purchases at $3.49 than at $4.49 (32% vs. 23%). But if one bottle costs more than about $1.16 to produce, $4.49 may generate more profit per shopper. I do not have the production cost, so I cannot confidently choose the final price yet.

**What changed my thinking.** Run 5. I first thought the lowest price was best, because price had the clearest effect. My profit calculation showed that higher sales do not always mean higher profit. Run 9 also made me less confident about online.

**Next experiment.** One online experiment, with shoppers randomly assigned to four versions: $3.49 or $4.49, each with the environment or cleaning message. About 3,700 shoppers if results match my model, about 15,000 if the differences are smaller. I would measure first purchases and eight-week repeat purchases.

**What would change my decision.** If $4.49 sells almost as well as $3.49, or $3.49 buyers rarely buy again, I would choose $4.49. If cleaning beats environment, I would switch messages. If the company shares its cost per bottle, the price may not need a separate test.

## Final recommendation in detail ( Word limit more than 200 words here)

### 1. Recommendation

I recommend testing the product online first, with the environment message as one of the options.

My model predicts that the product may get more purchases at $3.49 than at $4.49 (32% vs. 23%). However, a lower price does not always mean higher profit. If the cost of producing one bottle is above approximately $1.16, selling it at $4.49 may generate more profit per shopper. Since I do not have the actual production cost, I cannot confidently choose the final price yet.

### 2. What changed my thinking

Run 5 changed my thinking. Initially, I thought the lowest price was the best choice because price had the clearest effect on purchase decisions. However, my profit calculation showed that higher sales do not always mean higher profit.

Run 9 also made me less confident about online sales, because the online advantage became weaker after I removed visits with no shelf message.

### 3. Next experiment

I recommend running one online experiment with four versions:

- $3.49 with the environment message
- $3.49 with the cleaning message
- $4.49 with the environment message
- $4.49 with the cleaning message

I would randomly assign shoppers to these four versions to make the comparison fair. I estimate that around 3,700 shoppers may be needed if the actual results match my model's predictions, or around 15,000 if the differences are smaller than expected.

I would measure first-time purchases and repeat purchases within eight weeks.

### 4. What would change my decision?

If the $4.49 price achieves almost the same sales as $3.49, or if customers who buy at $3.49 rarely purchase again, I would consider choosing $4.49.

If the cleaning message performs better than the environment message, I would switch to the cleaning message.

If the company shares the actual cost per bottle, I may be able to make a better pricing decision without running a separate price experiment.

### 5. Model history

My model carries two pieces of information from one visit to the next: whether a household has bought the new cleaner before, and whether its feedback after using the product was positive or negative.

I also tested other history features, such as repeating the previous choice, but they did not improve the model enough to justify including them.

For more details about the model's limitations, failed ideas, and testing process, see [TRUST.md](TRUST.md).

### 6. How I estimated 3,700 and 15,000 shoppers (run 11)

I used the standard formula for comparing two groups, with 95% confidence (1.96) and 80% power (0.84):

```
shoppers per group = (1.96 + 0.84)^2 x [p1(1 - p1) + p2(1 - p2)] / (p1 - p2)^2
```

The hardest question in the test is the message, because its gap is the smallest:

- Environment message: p1 = 27.6% of shoppers buy (model, run 5)
- Cleaning message: p2 = 23.6%
- Gap = 4.0 points

```
shoppers per group = 7.85 x (0.276 x 0.724 + 0.236 x 0.764) / 0.040^2
                   = 7.85 x (0.200 + 0.180) / 0.0016
                   = about 1,863 per message group
```

- Two message groups: 2 x 1,863 = **about 3,700 shoppers** (about 930 per version).
- If real shoppers react only half as much (a 2-point gap): the gap is squared in the formula, so halving it needs 4 times more shoppers, 2 x 7,462 = **about 15,000 shoppers** (about 3,700 per version).
- Price has a bigger gap (8.6 points) and needs only about 400 per group, so the message question sets the test size.

Code: `experiments/e11_test_size.py`. Saved result: run 11 (`python runlog.py show 11`).
