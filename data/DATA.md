# Data Description

## Source
**Concrete Compressive Strength** dataset, UCI Machine Learning Repository
(https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength).
Donor / original study: I-Cheng Yeh, "Modeling of strength of high-performance concrete using
artificial neural networks", *Cement and Concrete Research*, 28(12), 1998.

The file is distributed as `.xls`; it was exported to `data/concrete.csv` without any other change.

## Task
Regression: predict the compressive strength of a concrete mix (a continuous value) from its
ingredients and age.

## Size
1,030 rows, 8 input features, 1 target. The original data has no missing values.
(The CSV loader still drops any row with a missing or non-numeric cell and reports the count.)

## Columns

| # | Column | Unit | Role |
|---|---|---|---|
| 1 | Cement | kg per m³ of mixture | feature |
| 2 | Blast furnace slag | kg per m³ | feature |
| 3 | Fly ash | kg per m³ | feature |
| 4 | Water | kg per m³ | feature |
| 5 | Superplasticizer | kg per m³ | feature |
| 6 | Coarse aggregate | kg per m³ | feature |
| 7 | Fine aggregate | kg per m³ | feature |
| 8 | Age | days (1 to 365) | feature |
| 9 | Concrete compressive strength | MPa | **target (last column)** |

## Why it suits a small neural network
- The relationship between mix proportions, age and strength is nonlinear (for example, strength
  rises quickly with age at first and then levels off), so a hidden layer should beat a linear model.
- All columns are numeric, so no encoding is needed.
- Features have very different ranges (age in days vs. kg/m³ quantities), which makes the
  normalisation step genuinely necessary.

## Preprocessing used
1. Shuffle with a fixed seed (42), then split **70% train / 15% validation / 15% test**.
2. Z-score scaling `(x - mean) / std` of all features **and the target**, with mean and std
   computed from the **training rows only** (no information leaks from validation or test).
3. Predictions are converted back to MPa before any metric is computed.