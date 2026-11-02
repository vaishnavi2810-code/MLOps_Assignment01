# Baseline Results – `01_spam_tutorial.ipynb`

Original notebook, run end to end with no changes to the labeling functions (LFs).
These are the reference numbers every LF experiment is compared against.

**Run date:** 2026-11-01
**Environment:** lab `.venv`, Python 3.11 (first run on the repo-root Python 3.14 `.venv` gave identical numbers)
**Data:** 1,586 unlabeled training comments (videos 1–4), 250 labeled test comments (video 5, Shakira)

## Headline metrics

| Metric | Cell | Value |
|---|---|---|
| Majority Vote accuracy (test) | 97 | **83.6%** |
| Label Model accuracy (test) | 97 | **86.4%** |
| Logistic Regression trained on Label Model labels (test) | 115 | **93.2%** |

## LF summary on the training set (cell 88)

| LF | Polarity | Coverage | Overlaps | Conflicts |
|---|---|---|---|---|
| keyword_my | SPAM | 0.199 | 0.188 | 0.113 |
| keyword_subscribe | SPAM | 0.127 | 0.109 | 0.070 |
| keyword_http | SPAM | 0.119 | 0.101 | 0.083 |
| keyword_please | SPAM | 0.112 | 0.110 | 0.058 |
| keyword_song | HAM | 0.142 | 0.113 | 0.044 |
| regex_check_out | SPAM | 0.234 | 0.132 | 0.087 |
| short_comment | HAM | 0.226 | 0.151 | 0.074 |
| has_person_nlp | HAM | 0.130 | 0.098 | 0.044 |
| textblob_polarity | HAM | 0.035 | 0.033 | 0.005 |
| textblob_subjectivity | HAM | 0.358 | 0.265 | 0.160 |

Polarity: SPAM = 1, HAM = 0. Coverage, Overlaps and Conflicts are fractions of the 1,586 training comments.

## Observations

- 5 SPAM LFs and 5 HAM LFs.
- `textblob_subjectivity` has the highest coverage (35.8%) and also the most conflicts (16.0%), so it's probably the noisiest LF.
- `keyword_my` overlaps on almost every comment it covers (0.188 of 0.199) and conflicts on 11.3%. The single word "my" is probably too broad.
- `textblob_polarity` barely fires (3.5% coverage), so it contributes very little.
- The Label Model beats Majority Vote by 2.8 points, so weighting LFs by their estimated accuracy helps.
