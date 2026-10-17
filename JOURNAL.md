# Journal

A running log of what I looked at, what I thought, and what I decided. Newest stage at the bottom.

## Stage 0: Understand the data (before any code changes)

**Starter result.** Top choice matched 29.0% (build) and 28.0% (check). It predicts 15.9% entrant share while the actual share is 21.1%, so it undersells the new cleaner. Its weights are hand-set and learn nothing from visits.

**How were the levers assigned?** Entrant price, message and channel are close to balanced against each other, against household budget, and across visit numbers. Every household saw more than one price. So the data looks like a rough randomized experiment, which means raw comparisons are a fair first look. Two exceptions to check later:
- Blank messages are mostly in retail (103 retail vs 33 online). "Blank" may mean no shelf sign, or missing data. It is not random.
- Tight-budget households saw $3.49 a little more often (137 vs about 110). Could be chance.

**Raw entrant share (build set):**

| Lever | Level | Entrant share |
|---|---|---|
| Price | $3.49 / $4.49 / $5.49 | 29.4% / 19.4% / 13.3% |
| Message | cleaning / environment / low_odor / blank | 21.4% / 24.5% / 17.2% / 21.3% |
| Channel | online / retail | 23.7% / 18.5% |

The price effect is big and steady. The message and channel gaps are smaller. With only ~350 visits per message, spread over 200 households, they could be noise.

**Free text.** `said_why` is filled in for 24% of visits, `after_use_note` for 19%. The after-use notes for the entrant include "It left a stain behind" (13) more often than "It removed the stain I noticed" (8). That fits the product's own claim that old stains may need a second pass, so repeat purchase may be the real risk, not the first sale.

**Open questions for later stages:**
1. Are message and channel effects real once we account for six visits per household?
2. Does buying the entrant once make a household more or less likely to buy it again?
3. Does buying anything make "none" more likely on the next visit (the household still has cleaner at home)? The starter assumes so. Check it.
4. Is `said_why` a real reason or a story told after the fact? For example, do people say "less waste" mostly when the shelf said "environment"?
5. Which price is best for the brand? Lowest price wins share, but we do not know cost or margin.

**Decision.** Commit FIRST.md now, then run the household bootstrap on the build set. Do not touch `check_visits.csv` for decisions yet.

## Stage 1: Run log and the FIRST.md test

**Set-up.** `runlog.py` stores every run in `runs.sqlite`: the expectation (saved before the code runs), inputs, outputs, model version, data version (a hash of the CSV files) and code version (the git commit). Failed runs are kept. A verdict is added after reading the outputs. `python runlog.py list` and `python runlog.py show <id>`.

**Run 1: starter vs a "know nothing" model.** I expected the starter to be worse than always predicting the average shares. I was wrong: log loss 1.579 vs 1.594 on build. But the gain is only about 1%, so the hand-set weights carry little signal. Any new model has to clearly beat 1.58.

**Run 2: household bootstrap (the FIRST.md test).**

| Gap in entrant share | Points | 95% interval (households resampled) |
|---|---|---|
| $3.49 vs $5.49 | +16.1 | 11.2 to 21.1 |
| $3.49 vs $4.49 | +10.0 | 4.3 to 15.6 |
| $4.49 vs $5.49 | +6.1 | 0.9 to 11.4 |
| environment vs low_odor | +7.3 | 1.8 to 13.4 |
| environment vs cleaning | +3.1 | -3.3 to 9.5 |
| cleaning vs low_odor | +4.2 | -1.5 to 10.0 |
| online vs retail | +5.2 | 0.8 to 9.5 |

What I learned:
1. **Price is real.** Each $1 step costs the entrant 6 to 10 points of share.
2. **My clustering worry was wrong for gaps.** Resampling households gives the same width as resampling single visits. Households *do* differ in taste (spread of entrant buys per household is 1.58 vs 1.00 if all were alike), but each household saw several prices and messages, so its own taste cancels out inside a gap. Clustering still matters for overall levels, such as "the entrant's share is 21%".
3. **My FIRST.md rule turned out too loose.** By that rule, "environment beats low_odor" and "online beats retail" both pass. But I tested 7 gaps, and I chose environment vs low_odor because it was the biggest raw gap. Both lower ends sit near zero. Environment vs cleaning, the comparison that matters for choosing a message, is not clear at all.

I am flagging this openly instead of quietly changing the rule: by the rule as written they pass. My judgment is that the message and channel evidence is still weak.

**Choices barely carry over between visits.** A household repeats its last choice 23.1% of the time, vs 20.6% if visits were independent. So history may add little. The model stage will test this properly.

**Next.** (a) Fit a simple learned choice model on the build set and compare it with run 1. (b) Decide in advance, before looking, a stricter test for message and channel: one that controls for price and the other levers at the same time, and that counts the number of comparisons.
