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
*(Fill in after running `all`.)*

| Model | Test RMSE (MPa) | Test MAE (MPa) | Test R² |
|---|---|---|---|
| Predict training mean | | | |
| Linear regression | | | |
| Neural network (best config) | | | |

**Selected configuration:** learning rate = , hidden nodes = , epochs = , momentum = , weight decay = .

## Analysis
*(Write after viewing the charts. Suggested points.)*
- **Learning rate:** too low underfits within the epoch budget; too high oscillates or diverges
  (cells marked "div." in the heatmap).
- **Hidden nodes:** few nodes underfit; many nodes lower the training error but may widen the
  train/validation gap (`overfit_gap` column).
- **Epochs:** compare the training and validation curves in `epochs_effect.png` and `loss_curves.png`.
- **Momentum and weight decay:** compare `sweep_results.csv` against the grid winner.
- **Seed variance:** is the winner's advantage larger than the standard deviation across seeds?
- **Residuals:** any pattern (for example, larger errors at high strength) points to model limits.