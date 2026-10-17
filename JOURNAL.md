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

## Stage 2: A model that learns, built one piece at a time

**What I built.** `choice_model.py`. Same idea as the starter: each option gets a score, and scores turn into chances. The difference is that the model learns the size of each effect from the build visits instead of me guessing them. I can switch pieces on and off, so I can add one idea at a time and see if it actually helps.

**How I judge "helps".** I split the 200 build households into 5 groups. The model learns from 4 groups and gets tested on the 5th, which it has never seen, and I repeat that 5 times. Lower log loss is better. I still haven't touched the check set.

**Run 3: adding pieces one by one**

| Model | What it knows | Log loss (lower is better) |
|---|---|---|
| Starter | hand-set guesses | 1.579 |
| A | brand + price | 1.587 |
| B | A + usual brand + budget | 1.575 |
| B + message | | 1.574 |
| B + channel | | 1.575 |
| C | B + message + channel | 1.574 |

- Price: learned as -0.45, close to the starter's -0.4. The starter's guess there was fine.
- Usual brand matters (+0.49). But it helped much less than I expected (I guessed 1.55).
- **Message and channel failed the rule I wrote down before the run.** Message p = 0.057, channel p = 0.026, and I had set p < 0.01 as the bar. Adding them barely changes the test score.
- **The big lesson:** every model sits around 1.57 to 1.59. Most of what a household picks on a given visit just can't be predicted from these fields. That's worth saying plainly to the brand.

**Run 4: does the past help?**

| History piece | Weight | Clear? | Visits it applies to |
|---|---|---|---|
| bought the entrant before | +0.56 | yes (z = 3.4) | 442 |
| bad note after using the entrant | -0.82 | maybe (z = -1.9) | 51 |
| good note after using the entrant | +0.20 | no | 76 |
| same choice as last visit | 0.00 | no | 823 |
| buy nothing right after buying | +0.08 | no | 823 |

- History helps a little (1.574 to 1.570), and nearly all of the gain is "bought the entrant before". A raw count agrees: 27.1% buy it again vs 17.5% for households that never bought it.
- I was wrong about "same as last visit": it's zero, not +0.15.
- The starter assumes people buy nothing right after buying, because they still have cleaner at home. The data doesn't back that up.
- **What I can't tell:** maybe trying the entrant made people like it, or maybe some households just liked it from the start. Both look the same in this data. A real test would need first-time buyers tracked over time.

**What the model carries from one visit to the next:** whether the household bought the entrant before, and whether it left a good or bad note after using it. Nothing else.

**Checks I added.** A test that changing a later visit never changes what the model knew at an earlier visit (no peeking at the future), plus basic tests that the chances add up to 1 and that a higher price lowers the entrant's chance.

**Where I stand now.**
- Price: clear. Lower sells more.
- Message: not clear. Environment looks best, but it doesn't pass my bar.
- Channel: not clear. Online looks a bit better, but it doesn't pass either.
- Repeat buying: the entrant has some pull on people who tried it, and bad stain notes may hurt it.

**Next.** Turn the model into real numbers for the brand (expected entrant share for each price, message and channel combination, with uncertainty). Look at `said_why` to see if people's stated reasons can be trusted. Only after that, run the model once on the check set.
