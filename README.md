# AI-Powered Student Analytics and Success Platform

Early-warning system that spots students who may struggle academically, explains why, and suggests what support to give, before marks drop.

> Built for a 24-hour hackathon (Challenge 04). All data is **synthetic**; no real student information is used.

## Problem
Institutions collect attendance, marks, assignment and engagement data, but usually notice struggling students only after their performance has already declined.

## Our solution
Student data → preprocessing → ML risk model → explanation → suggested intervention

- **Risk prediction:** Low / Moderate / High for every student
- **Explainable:** top reasons behind each student's risk (e.g. low IA marks, low LMS engagement, falling attendance)
- **Early-warning signals:** attendance trend, IA change, LMS activity, assignment submission
- **Suggested action:** remedial class, mentor check-in, counselling referral, etc.
- **Fairness check:** compare results across gender, area, first-gen and income groups (not used as model inputs)

## Dataset (synthetic, 100 students, 8 linked tables)
`students_master`, `semester_performance`, `course_marks`, `weekly_engagement`, `wellbeing_survey`, `activities`, `interventions`, `journey_events`
Includes 8 hidden student types (e.g. Silent Decliner, Burnout Risk, Late Bloomer) to test pattern discovery.

## Model
- Random Forest (300 trees, class-balanced), 30 numeric features
- Labels, answer-key and fairness columns are **not** model inputs (no leakage)
- 5-fold cross-validation on 98 active students

| Metric | Result |
|---|---|
| Accuracy | 78% |
| High-risk recall | 94% (16 of 17) |
| High-risk students missed as Low | 0 |

**Top risk drivers:** IA marks, LMS engagement, attendance, assignment scores.

> Small synthetic sample, so numbers are indicative, not production-ready.

## Files
- `ml_training.ipynb`: training and evaluation steps
- `risk_model.pkl` and `features.json`: trained model and its 30 input columns

## Use the saved model
```python
import joblib, json, pandas as pd
model = joblib.load("risk_model.pkl")
cols = json.load(open("features.json"))
new = pd.read_csv("students_master.csv")[cols]
print(model.predict(new))
