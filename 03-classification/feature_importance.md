# Feature Importance

Before training a model it helps to know which features actually matter for
predicting the target (here: `churn`). There are three common ways to
measure this, and each fits a different type of feature:

| Method | Feature type | Question it answers |
|---|---|---|
| Risk ratio / Risk difference | Categorical | Does this *group* churn more or less than average? |
| Mutual information | Categorical | How much does knowing this feature reduce uncertainty about churn? |
| Correlation | Numerical | Does this *value* move up/down together with churn? |

---

## 1. Risk Ratio & Risk Difference (categorical features)

For each category (group) of a categorical column, compare its churn rate
to the **global churn rate** (the churn rate across the whole dataset).

```python
global_churn_rate = df_full_train.churn.mean()

df_group = df_full_train.groupby("gender").churn.agg(['mean', 'count'])
df_group['risk'] = df_group['mean'] / global_churn_rate
df_group['difference'] = df_group['mean'] - global_churn_rate
df_group
```

- **Risk ratio** = `group churn rate / global churn rate`
  - `risk > 1` → this group churns **more** than average → higher risk
  - `risk < 1` → this group churns **less** than average → lower risk
  - `risk == 1` → this group behaves like the average, feature not very
    informative for this group
- **Risk difference** = `group churn rate - global churn rate`
  - Same idea, but expressed as an absolute difference (in percentage
    points) instead of a ratio. Easier to compare magnitudes across groups.

Run it for every categorical column to see which groups deviate the most
from the baseline:

```python
for column in categorical:
    df_group = df_full_train.groupby(column).churn.agg(['mean', 'count'])
    df_group['risk'] = df_group['mean'] / global_churn_rate
    df_group['difference'] = df_group['mean'] - global_churn_rate
    display(df_group)
```

**Limitation:** this only tells you about *individual groups* within a
feature (e.g. "month-to-month contracts are risky"), not an overall score
for the whole feature. For that, use mutual information.

---

## 2. Mutual Information (categorical features)

Mutual information (MI) measures how much information one variable gives
you about another — i.e. how much uncertainty about `churn` is removed
once you know the value of a feature. It gives a **single score per
feature**, unlike risk ratio/difference which is per-category.

```python
from sklearn.metrics import mutual_info_score

def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)

mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending=False)
```

- MI = 0 → the feature and `churn` are independent (knowing the feature
  tells you nothing about churn)
- Higher MI → the feature is more informative about churn
- There's no fixed upper bound, so MI is best used to **rank** features
  against each other, not to interpret in isolation.

In the churn dataset, `contract` typically comes out on top, while
`gender` is near the bottom — consistent with what the risk table shows.

---

## 3. Correlation (numerical features)

Mutual information and risk ratio work on categorical (discrete) data.
For **numerical** features (`tenure`, `monthlycharges`, `totalcharges`),
use the **Pearson correlation coefficient** between the feature and the
(0/1-encoded) target.

```python
df_full_train[numerical].corrwith(df_full_train.churn)
```

- Correlation ranges from **-1 to 1**.
- **Sign** tells you the direction of the relationship:
  - Positive → as the feature increases, churn tends to increase
  - Negative → as the feature increases, churn tends to decrease
- **Magnitude** tells you the strength:
  - `|corr|` close to 0 (≈ 0.0–0.2): weak relationship
  - `|corr|` moderate (≈ 0.2–0.5): moderate relationship
  - `|corr|` close to 1 (≈ 0.5+): strong relationship

For churn: `tenure` is usually **negatively** correlated (longer-tenured
customers churn less), while `monthlycharges` is usually **positively**
correlated (customers paying more churn more).

---

## Summary

- **Risk ratio / difference** → per-category, easy to read, good for
  spotting *which specific groups* are high/low risk.
- **Mutual information** → per-feature score for *categorical* variables,
  good for ranking features by overall importance.
- **Correlation** → per-feature score for *numerical* variables, good for
  ranking features and knowing the direction of the relationship.

Together these three give a quick, model-free first pass at feature
importance before doing any actual model training (e.g. logistic
regression).
