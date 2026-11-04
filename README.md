# Neural Network Regression in C# (from scratch)

A single-hidden-layer neural network (input → tanh hidden → linear output) written in plain C#,
trained with back-propagation, and applied to the UCI **Concrete Compressive Strength** dataset.
The network engine uses no machine-learning libraries. ScottPlot is used only to draw charts.

## Requirements
- .NET 10 SDK (works on Linux, Windows, macOS)
- Internet access on first build (restores the ScottPlot NuGet package)

## Setup
1. Download the dataset from https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
2. Open the `.xls` in a spreadsheet program and save it as `data/concrete.csv`
   (comma-separated, header row optional, **target in the last column**).

## Run
From the repository root:

```bash
dotnet build src/NeuralNetRegression
dotnet run --project src/NeuralNetRegression -- baseline   # quick sanity check
dotnet run --project src/NeuralNetRegression -- tune       # grid search + sweep (validation only)
dotnet run --project src/NeuralNetRegression -- final      # test-set evaluation of the winner
dotnet run --project src/NeuralNetRegression -- all        # tune, then final
```
An optional second argument gives a different CSV path.

If PNG export fails on Linux with a missing native library, add
`SkiaSharp.NativeAssets.Linux.NoDependencies` to the project.

## Project layout
```
src/NeuralNetRegression/
├── Program.cs                      command-line dispatch
├── Experiments.cs                  baseline / tune / final workflows
├── Network/NeuralNetwork.cs        forward pass, back-propagation, momentum, weight decay
├── Data/CsvLoader.cs               CSV parsing, header detection, missing-value handling
├── Data/DataSplitter.cs            seeded 70/15/15 split
├── Data/StandardScaler.cs          z-score scaling fitted on training rows only
├── Data/PreparedData.cs            load + split + scale in one step
├── Evaluation/Metrics.cs           MSE, RMSE, MAE, R²
├── Evaluation/ModelEvaluator.cs    scaled predictions → original units → metrics
├── Evaluation/LinearRegressionBaseline.cs   OLS reference model
├── Tuning/HyperparameterTuner.cs   parallel grid search over seeds
├── Tuning/ResultsIo.cs             CSV outputs
└── Plotting/Plots.cs               ScottPlot charts
data/                               concrete.csv and DATA.md
results/                            logs, tables and charts written by the program
```

## Method
**Forward pass.** `h = tanh(W1·x + b1)`, `y = W2·h + b2` (identity output, so predictions are
unbounded real numbers).

**Loss.** Per sample `E = ½ Σ (t − y)²`, reported as MSE / RMSE.

**Back-propagation.**
- Output signal: `δ_o = t − y` (linear output, so there is no activation derivative).
- Hidden signal: `δ_h = (1 − h²) · Σ δ_o · w_ho` (tanh′ = 1 − h²), computed before the weights change.
- Update: `w += lr · (δ · input − weightDecay · w) + momentum · previousDelta`.
  Signals are `target − output`, i.e. the negative gradient, hence `+=`. Weight decay is applied to
  weights only, not biases.

**Initialisation.** Uniform in ±1/√fanIn, biases zero, seeded.

**Preprocessing.** Fixed-seed shuffle; 70/15/15 split; features **and** target z-scored using
training statistics only; metrics are computed after converting predictions back to MPa.

## Evaluation protocol
- **Validation set** is used for all tuning decisions. **Test set** is used once, in `final`.
- Every configuration is repeated with 5 seeds and reported as mean ± std.
- Reference points: predict-the-training-mean and ordinary linear regression.
- Metrics: RMSE, MAE, R². Charts: loss curves, predicted vs. actual, residuals, tuning heatmap,
  effect of epochs.

## Tuning design
| Stage | What varies | Values |
|---|---|---|
| Grid | learning rate | 0.001, 0.005, 0.01, 0.05, 0.1 |
| | hidden nodes | 2, 4, 8, 16, 32 |
| | epochs | 100, 500, 1000, 2000 |
| Sweep (around the grid winner) | momentum | 0, 0.5, 0.9 |
| | weight decay | 0, 1e-4, 1e-3 |

100 grid configurations × 5 seeds = 500 results. Because training is deterministic per seed with no
epoch-dependent schedule, one 2000-epoch run is read at epochs 100/500/1000/2000, which is
equivalent to training separately for each epoch count. The grid uses momentum 0.5 and no weight decay.

## Outputs (`results/`)
| File | Content |
|---|---|
| `training_log_baseline.csv` | per-epoch loss of the baseline run |
| `tuning_results.csv` | all 100 grid configurations (mean ± std over seeds, overfit gap) |
| `sweep_results.csv` | momentum × weight-decay sweep |
| `best_config.csv` | selected hyperparameters |
| `training_log.csv` | per-epoch loss of the final model (epoch, train/validation MSE and RMSE) |
| `final_summary.csv`, `predictions_test.csv` | test metrics and per-sample predictions |
| `*.png` | all charts |

## Results

| Model | Test RMSE (MPa) | Test MAE (MPa) | Test R² |
|---|---|---|---|
| Predict training mean |16.66 | 13.58 | -0.020 |
| Linear regression | 11.77 | 8.83 | 0.491 |
| Neural network (best config) | 5.58 ± 0.17 | 4.13 | 0.885 |

The neural network figures are means over seeds. The ± is the standard deviation of test RMSE.

**Selected configuration:** learning rate = 0.01, hidden nodes = 16, epochs = 500, momentum = 0.5, weight decay = 0.0001.

## Discussion

- **Learning rate.** The best results sit in the 0.005 to 0.01 columns (about 5.9 to 6.6 RMSE for 4 or more nodes). At 0.001 the network is probably still underfitting within 500 epochs. At 0.05 error rises (6.7 to 9.1), and at 0.1 it is poor everywhere (8.9 to 9.7). The 32-node network at 0.1 diverges outright. Higher learning rates fail harder as width grows.

- **Hidden nodes.** 2 nodes clearly underfit (7.6 to 9.7). Error drops sharply by 4 to 8 nodes and is best at 16 (5.93 at lr 0.01). At 32 nodes it gets worse again (6.5 to 6.8 at the same learning rates), so extra capacity does not help here. I can’t say whether that is overfitting or unstable training without the overfit_gap column.

- **Epochs.** The validation curve drops fast over the first ~60 epochs, then flattens and keeps improving slowly, reaching about 5.5 to 5.7 by epoch 500. Training RMSE ends near 4.0, a gap of about 1.5 MPa. There is no sign of validation error turning upward, so early stopping isn’t needed. The still-falling curve suggests more epochs might help slightly. The validation curve is noisy (swings of about ±0.3), which is typical of stochastic updates with momentum.

- **Momentum and weight decay.** These weren’t in the grid, so compare the winner against sweep_results.csv. Add that comparison here.

- **Seed variance.** The winner’s margin over its neighbors is small: 5.93 versus 5.97 (lr 0.005, 16 nodes), 6.01 (lr 0.001, 32 nodes) and 6.06 (lr 0.005, 8 nodes). That is well inside the ±0.17 seed standard deviation. So the best config is really a plateau of roughly equivalent models rather than a clear winner. The gap to the baselines is not ambiguous: 5.58 versus 11.77 for OLS is about 35 standard deviations.

- **Residuals.** Residuals. The residuals scatter around zero across the whole range of predicted strength (about 3 to 80 MPa), with no curve or trend. That suggests the network has captured the main nonlinear structure and is not systematically biased at either end. The spread is roughly constant (mostly within ±10 MPa), with no clear funnel shape, so there is little evidence of heteroscedasticity.

- **Overall.** The network explains about 89% of the variance in concrete strength, against 49% for linear regression. This shows the relationship is substantially nonlinear. The test RMSE (5.58) is a bit lower than the grid’s validation RMSE (5.93), which is plausible since they are different data splits.