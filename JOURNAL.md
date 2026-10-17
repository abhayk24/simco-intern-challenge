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
