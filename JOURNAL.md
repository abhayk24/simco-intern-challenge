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

## Stage 3: Numbers for the brand, stated reasons, and the one check-set test

**Run 5: what each option would sell on a first visit**

I asked model D: if a household sees the entrant for the first time at this price, message and channel, what's the chance it buys? Then I redid it 200 times on resampled households to see how much the answer moves.

Top options (entrant share, 90% range):
- $3.49 / environment / online: 32% (25% to 38%)
- $3.49 / cleaning / online: 28% (22% to 34%)
- $3.49 / environment / retail: 25% (20% to 31%)
- ...
- $5.49 / low_odor / retail: 8% (6% to 11%), the worst

How often each one comes out on top across the 200 redos:
- **Price:** $3.49 sells the most every single time.
- **Channel:** online beats retail 98.5% of the time.
- **Message:** environment is best 81.5% of the time, cleaning 17.5%, low_odor 1%.

**The catch on price.** Selling the most bottles isn't the same as making the most money. We don't know what a bottle costs the brand. So I worked out where the answer flips:
- If a bottle costs less than about $1.16: $3.49 makes the most money.
- Between about $1.16 and $2.36: $4.49 makes more.
- Above about $2.36: $5.49 makes more.

The range on that $1.16 is wide ($0.11 to $1.79). **The data can't pick the price for profit without the brand's cost.**

**A result that looks like a conflict.** Run 3 said messages are *not proven* different (p = 0.057). Run 5 says environment wins 81.5% of the time. Both are true. They answer different questions: "are the messages different at all?" vs "which one is most likely best?". Environment is the best bet, but a 1-in-5 chance it isn't is too high to call it settled.

**Run 6: can we trust what people say?**

I checked each `said_why` answer against facts we already have.
- Only 7.8% of the checkable answers flat-out contradict the facts. That's fewer than I expected.
- **But the bigger finding:** 15 of the 16 reasons always come with *one* product. "I know this brand" always goes with standard. "I wanted less smell" always goes with the entrant. Knowing the reason tells you the choice 94% of the time. So these answers mostly repeat each product's own claims. They aren't an independent "why".
- Where answers do contradict facts:
  - "I bought the cleaner I often buy" disagrees with the survey's usual brand 9 out of 31 times.
  - "I still had cleaner at home" came from people who hadn't bought anything the visit before, 7 out of 18 times.
- I guessed the "environment" sign would make people *say* "less waste" more often. It didn't (36% vs 46% and 50%). Small numbers, though.
- Entrant buyers answer more often than others (29% vs 21%), so the answers aren't a fair sample either.
- **Decision:** I won't use `said_why` as a model input, or as the "reason" the app shows.

(Side note: I ran the one-reason-one-product count on the check set too, just to see whether the pattern holds. It does (96%). No model decision used it.)

**Run 7: the one-time test on the check set**

I locked model D (trained on all 200 build households) and wrote the pass rules first. All three passed:

| Model | Log loss on check (lower is better) |
|---|---|
| D (my model) | 1.552 |
| C (no history) | 1.554 |
| Starter | 1.563 |
| Always-average | 1.591 |

- D beats the starter, but only by 0.011, and the range (-0.038 to +0.015) includes zero. So: better on average, not proven better on 50 households.
- **D slightly oversells the entrant**: it predicts 21.0% and the truth is 18.3%. In the middle group (where D says 15-25%), only 13% actually buy.
- On new households, price holds strongly: 25.5% at $3.49 vs 7.7% at $5.49.
- Message and channel point the same way as before (environment 21.9% vs 17.9% for the others; online 22.2% vs 14.3% retail). That supports them, but at about 100 visits per message it can't settle anything.

**The check set is now used.** If I change the model after this, a new check-set score won't count as a fresh test, and I'll say so.

**Where I lean right now (not final):**
- **Channel:** online. Clear in build, and it held on new households.
- **Message:** environment. The most likely best, not proven.
- **Price:** I don't know yet. It depends on the brand's cost per bottle. $3.49 sells the most, but if a bottle costs more than about $1.16, $4.49 makes more money.

**Next.** Build the small live app (change an input, get a new answer), with household-level reasons that show their evidence. Then the remaining trust checks (what fails, what the data can't tell us), the real-shopper test design, and the final note.

## Stage 4: The live app

**What I built.** `app.py` is a small web page that runs with just Python (no installs). You pick a household from the data, or describe a new one. You change the new cleaner's price, message, channel or the other cleaners' prices. You get back:
- the chance of each choice, plus a likely range for the new cleaner (from 100 refits on resampled households, saved in run 8);
- **"change one thing" reasons** for that household: switch one piece off and see how many points the new cleaner's chance moves;
- a **trust label** on each reason ("solid", "weak" or "not proven"), plus which saved run backs it or doubts it;
- what the chance would be at $3.49, $4.49 and $5.49;
- the household's past visits. `said_why` is shown, but marked as not used (run 6);
- the model's limits, always on screen.

The same answer is available as JSON at `/api/predict`. All saved runs can be opened at `/runs/<id>`.

**Why "change one thing" reasons.** The starter gave one fixed sentence. I wanted a reason you can check: if the page says "online adds 5.5 points", you can switch the channel to retail and see the number drop. The trust label stops a big but shaky piece from looking like a fact. For example, a bad note after use can move the chance by -15 points, but it rests on only 51 visits, so it shows as "not proven".

**Choices I made:**
- The model is trained once and saved (`model/model_D.json`), so the app starts fast and every answer names its model and data version.
- Prices outside what the data covers still work, but show a warning that it's a guess past the data.
- Income, spend, bathrooms and the survey note aren't used, and the page says so. I never tested them.

**Checks.** 9 new tests: a changed input gives a new result; GET and POST give the same answer; bad inputs get a clear error; every reason has a trust label and evidence; the likely range contains the main answer; all the pages load.

## Stage 5: Last trust checks and the answer

**Run 9: three more checks.**
- **Shuffle test for message.** Fake message gaps were as big as the real one 9.5% of the time. Message is still not proven.
- **Online without blank-message visits.** Blank messages sit mostly in retail. Without them, online weakens (+0.33 to +0.27, z 2.2 to 1.7). I'm less sure about online than I was.
- **Surprise: who the shopper is matters more than the sign.** Households whose survey note sounds green ("go down the drain", "fewer plastic bottles") buy the new cleaner a lot more (+0.82, z = 4.1, 41 households). But the environment *sign* adds nothing extra for them. They like the product anyway.

**Run 10: is "bought before" just green households?** Only a little. "Bought before" drops from +0.56 to +0.52 and stays clear. So "taste or habit?" stays open. The green note is the single most useful piece I found (cross-validation 1.570 to 1.565). But I found it after the check set was used up, so I kept it out of the app on purpose. It goes into the real-shopper test instead.

**Run 11: how big must a real test be?** A 4-version online test ($3.49 or $4.49, each with environment or cleaning) needs about 3,700 shoppers if real gaps match the model, and about 15,000 if they're half as big. The message question, not price, sets the size.

**The answer, in one line each:**
- **Channel: online.** A lean, not proof. It held on the check set but weakened in run 9. It's also the easiest place to show shoppers random versions.
- **Message: environment as the lead,** tested against cleaning, because the two are not proven apart.
- **Price: I don't know yet.** It depends on the cost per bottle (run 5). Test $3.49 against $4.49, or just ask the brand for its cost.

**The run that changed my mind most: run 5.** Before it, I thought "lowest price wins". After it, I saw that the clearest finding in the data (price) still doesn't make the business decision on its own.

**A flaw I found in my own run log (final check).** Each run saves the git commit it ran on. Two records are misleading:
- **Run 1** says `aa3dba9`, but `runlog.py` and `e01_baseline.py` were brand-new files that weren't committed yet. My "uncommitted changes" check ignored new files. They were committed right after the run, in `8c5d112`.
- **Run 4** says `+dirty`, but the only uncommitted change was `runs.sqlite` (the log itself, holding run 3's verdict), not code. The e04 code is exactly commit `daa72d1`.

I fixed the check for future runs: it now counts new code files and ignores the log's own files. I left the old records as they are and explain them here, because editing saved runs would defeat the point of the log.

**Added after the final check: `experiments/e00_explore.py`.** The Stage 0 counts (purchase rates by price, message and channel, the fairness cross-tables, the free-text counts), and four quick checks quoted later (spread 1.58, repeat 23.1%, 27.1% vs 17.5%, said_why 94%), were first run as one-off snippets and never saved. This script reproduces all of them exactly, so every number in this journal can be traced to code. It only reads data and prints, so it isn't a saved run.
