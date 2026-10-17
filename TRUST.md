# Can the brand trust this system?

Short answer: trust it on **price direction**. Treat **message** and **channel** as leads to test, not facts. Don't trust any of its **exact numbers** for real US shoppers.

## Checks I built

Every run wrote its expectation down before running (`python runlog.py show <id>`).

| Check | Run | What it showed |
|---|---|---|
| Starter vs "always predict the average" | 1 | The starter beats it by only ~1%. The bar to beat was low. |
| Resample whole households, not single visits | 2 | Price gap is real. Message and channel gaps are near zero at the low end. |
| Cross-validation on build, split by household | 3, 4 | Each piece kept only if it helps on unseen households. |
| Strict rule for message/channel, written down first | 3 | Both failed it (message p=0.057, channel p=0.026, bar was p<0.01). |
| Fact-check stated reasons | 6 | Reasons mostly repeat each product's claims, so they aren't used. |
| One-time test on the 50 check households | 7 | Passed all 3 pre-set rules. Only slightly better than the starter. Oversells the new cleaner by ~3 points. |
| Shuffle test for message | 9 | Fake message gaps were as big as the real one 9.5% of the time. |
| Channel without blank-message visits | 9 | Online weakens: +0.33 → +0.27, z 2.2 → 1.7. |
| No peeking at the future | tests | Changing a later visit never changes what the model knew earlier. |

## Ideas that failed, conflicted, or stay untested

**Failed (my guess was wrong):**
- **Clustering (FIRST.md):** I expected six visits per household to widen the intervals. They didn't, for gaps, because each household saw several prices (run 2).
- **Repeating last choice:** I expected households to repeat their last choice. Effect: zero (run 4).
- **The starter's "buy nothing right after buying" idea:** not supported (run 4).
- **Stated reasons as the "why":** they echo the product's own claims and tell you the choice 94% of the time (run 6).
- **Shelf sign shaping stated reasons:** I expected the environment sign to make people *say* "less waste" more. It didn't (run 6).

**Conflicting:**
- **Message.** Not proven different (run 3, p=0.057; run 9, shuffle 9.5%). Yet environment is the best message in 81.5% of refits (run 5). Both are true: environment is the best bet, but the chance it isn't (about 1 in 5) is too high to call it settled.
- **Channel.** Online looks better in build (run 3), weaker once blank-message visits are dropped (run 9), and stronger on the check set (22% vs 14%, run 7).

**Untested or open:**
- **Taste or habit?** Households that bought the new cleaner before buy it again (+0.56). Is that habit, or did they like it all along? The green note explains only a little of it (run 10). Still open.
- **The green survey note** is the strongest new piece (+0.82, z=4.1, run 9), and it improves cross-validation more than any other single piece (run 10). But I found it after the check set was used up, so it has had no fresh test. It's **not** in the app.
- **Fields I never tested:** income, cleaning spend, bathrooms, message × channel, price × message.

## What the data cannot tell you

1. **Profit.** We don't have the cost per bottle. Below about $1.16, $3.49 makes the most money; between about $1.16 and $2.36, $4.49 does (run 5).
2. **Real shoppers.** Every household and answer is fictional. Real US shoppers will differ in size of effect, maybe in direction too.
3. **Why anyone chose.** Stated reasons aren't independent (run 6), and the survey brand often disagrees with what people say.
4. **Whether trying it causes coming back** (run 4 and run 10).
5. **What a blank message means.** No sign at all, or missing data? It's mostly in retail.
6. **Anything beyond 6 visits**, competitor price reactions, shelf position, or stock-outs.

## A real US shopper test that could show the system is wrong

**Design.** One online test. Each shopper who views the toilet-cleaner category is randomly shown one of 4 versions of the new cleaner:
- $3.49 + environment
- $3.49 + cleaning
- $4.49 + environment
- $4.49 + cleaning

The other cleaners stay at their usual prices. low_odor and $5.49 are left out (run 5).

**Size (run 11).**
- **About 3,700 shoppers** (~930 per version) if real gaps match the model.
- **About 15,000** (~3,700 per version) if real shoppers react half as much.
- Message, not price, sets the size.

**Measure:**
- first purchase of the new cleaner;
- repeat purchase within 8 weeks among buyers;
- one optional checkout question about plastic/drain worries, to test the green-note finding.

**The system is wrong if:**
- $4.49 sells as well as or better than $3.49 (the model says $3.49 wins by about 8.6 points); or
- the real share falls outside the model's 90% range in 3 or more of the 4 versions:

  | Version | Model's 90% range |
  |---|---|
  | $3.49 + environment | 25–38% |
  | $3.49 + cleaning | 22–34% |
  | $4.49 + environment | 18–28% |
  | $4.49 + cleaning | 16–24% |

**What would change my view:**
- **Cleaning beats environment** → switch the message.
- **The gap between $3.49 and $4.49 is under 3 points** → go with $4.49 (more money per bottle, few lost sales).
- **$3.49 buyers repeat much less than $4.49 buyers** (deal-seekers who don't stay) → go with $4.49.
- **Green-minded shoppers don't buy more** → drop the run 9 finding.
