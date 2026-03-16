# Dataset Preprocessing Report

## Filtering Pipeline

| Step | Description | Pairs Remaining |
|------|-------------|----------------:|
| 01 | Loaded from zip | 331,266 |
| 02 | Non-empty pairs | 331,266 |
| 03 | English length (2-12 words) | 323,310 |
| 04 | German length (2-15 words) | 323,024 |
| 05 | No URLs | 323,023 |
| 06 | No emails | 323,022 |
| 07 | No heavy numeric (>30% digits) | 323,019 |
| 08 | Length ratio (de ≤ 2.5x en) | 322,936 |
| 09 | Remove duplicates | 322,882 |
| 10 | Character filter (≥85% clean) | 322,376 |
| 11 | Vocabulary sanity | 321,703 |
| 12 | Capped at 20000 | 20,000 |

## Final Split Sizes

| Split | Size |
|-------|-----:|
| Train | 16,000 |
| Val   | 2,000 |
| Test  | 2,000 |
| **Total** | **20,000** |

## Sentence Length Statistics

| Metric | English | German |
|--------|--------:|-------:|
| Average words | 6.2 | 6.2 |
| Max words     | 12 | 15 |
| Min words     | 2 | 2 |

## Top 50 English Tokens

| Rank | Token | Count |
|-----:|-------|------:|
| 1 | `.` | 12,847 |
| 2 | `'` | 5,655 |
| 3 | `you` | 4,996 |
| 4 | `i` | 4,522 |
| 5 | `tom` | 3,980 |
| 6 | `to` | 3,432 |
| 7 | `?` | 3,211 |
| 8 | `the` | 3,109 |
| 9 | `t` | 2,352 |
| 10 | `a` | 2,137 |
| 11 | `is` | 1,736 |
| 12 | `that` | 1,695 |
| 13 | `do` | 1,281 |
| 14 | `s` | 1,250 |
| 15 | `it` | 1,217 |
| 16 | `in` | 980 |
| 17 | `he` | 942 |
| 18 | `have` | 940 |
| 19 | `of` | 927 |
| 20 | `me` | 901 |
| 21 | `was` | 878 |
| 22 | `this` | 857 |
| 23 | `we` | 823 |
| 24 | `,` | 812 |
| 25 | `don` | 796 |
| 26 | `what` | 773 |
| 27 | `for` | 757 |
| 28 | `are` | 712 |
| 29 | `can` | 687 |
| 30 | `be` | 666 |
| 31 | `your` | 647 |
| 32 | `did` | 612 |
| 33 | `my` | 606 |
| 34 | `mary` | 597 |
| 35 | `m` | 583 |
| 36 | `know` | 572 |
| 37 | `want` | 508 |
| 38 | `like` | 506 |
| 39 | `not` | 470 |
| 40 | `on` | 464 |
| 41 | `his` | 462 |
| 42 | `and` | 460 |
| 43 | `re` | 457 |
| 44 | `with` | 454 |
| 45 | `at` | 436 |
| 46 | `how` | 433 |
| 47 | `here` | 415 |
| 48 | `ve` | 411 |
| 49 | `think` | 409 |
| 50 | `ll` | 391 |

## Top 50 German Tokens

| Rank | Token | Count |
|-----:|-------|------:|
| 1 | `.` | 12,175 |
| 2 | `,` | 4,457 |
| 3 | `ich` | 4,337 |
| 4 | `tom` | 3,847 |
| 5 | `?` | 3,204 |
| 6 | `nicht` | 2,309 |
| 7 | `ist` | 2,157 |
| 8 | `du` | 2,033 |
| 9 | `das` | 1,997 |
| 10 | `sie` | 1,887 |
| 11 | `zu` | 1,406 |
| 12 | `es` | 1,310 |
| 13 | `die` | 1,234 |
| 14 | `hat` | 934 |
| 15 | `ihr` | 907 |
| 16 | `der` | 896 |
| 17 | `er` | 874 |
| 18 | `wir` | 840 |
| 19 | `in` | 795 |
| 20 | `ein` | 789 |
| 21 | `habe` | 772 |
| 22 | `!` | 767 |
| 23 | `was` | 717 |
| 24 | `mir` | 716 |
| 25 | `dass` | 663 |
| 26 | `wie` | 607 |
| 27 | `mit` | 578 |
| 28 | `mich` | 572 |
| 29 | `den` | 560 |
| 30 | `auf` | 554 |
| 31 | `war` | 554 |
| 32 | `sich` | 552 |
| 33 | `haben` | 548 |
| 34 | `eine` | 521 |
| 35 | `sind` | 458 |
| 36 | `noch` | 454 |
| 37 | `und` | 447 |
| 38 | `maria` | 447 |
| 39 | `hast` | 445 |
| 40 | `an` | 438 |
| 41 | `kann` | 411 |
| 42 | `bin` | 397 |
| 43 | `für` | 394 |
| 44 | `hier` | 385 |
| 45 | `so` | 384 |
| 46 | `dich` | 376 |
| 47 | `dir` | 373 |
| 48 | `einen` | 357 |
| 49 | `dem` | 355 |
| 50 | `sein` | 342 |

## 20 Random Cleaned Example Pairs

| # | English | German |
|--:|---------|--------|
| 1 | i suggest that you do that today. | ich schlage vor, daß sie das heute tun. |
| 2 | are you sure you want to swim here? | bist du sicher, daß du hier schwimmen willst? |
| 3 | how did you spend your holiday? | wie hast du deine ferien verbracht? |
| 4 | promise me you won't ever do that anymore. | versprecht mir, daß ihr das nie wieder tut! |
| 5 | do you have a bowling ball? | haben sie eine bowlingkugel? |
| 6 | i knew how tom felt. | ich wusste, wie tom zumute war. |
| 7 | this is a picture of the ship i was on. | das hier ist ein bild des schiffes, auf dem ich war. |
| 8 | the price was lower than i'd thought. | der preis war niedriger, als ich gedacht hatte. |
| 9 | burn it. | verbrennen sie es! |
| 10 | tom said he'd likely win. | tom hat gesagt, daß er wahrscheinlich gewinnen würde. |
| 11 | the rules should be followed. | die regeln sollten befolgt werden. |
| 12 | do you play any instruments? | spielen sie irgendwelche instrumente? |
| 13 | i milked the cows. | ich melkte die kühe. |
| 14 | the temperature in boston is in the lower 40s. | die temperatur in boston beträgt etwa fünf grad. |
| 15 | a massive locust outbreak continues threatening farmers’ fields in east africa. | eine gewaltige heuschreckenplage bedroht in ostafrika weiterhin die felder der bauern. |
| 16 | he became a cameraman after he graduated from college. | nach seinem universitätsabschluss wurde er kameramann. |
| 17 | tom was very gullible. | tom war sehr leichtgläubig. |
| 18 | let's get drunk. | betrinken wir uns! |
| 19 | pandas are beautiful animals. | pandas sind schöne tiere. |
| 20 | did you see the way tom was looking at you? | ist euch aufgefallen, wie tom euch angesehen hat? |

---
*Generated by `preprocess_tatoeba_en_de.py` with seed=42*
