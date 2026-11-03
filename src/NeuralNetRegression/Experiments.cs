using NeuralNetRegression.Data;
using NeuralNetRegression.Evaluation;
using NeuralNetRegression.Plotting;
using NeuralNetRegression.Tuning;

namespace NeuralNetRegression;

/// <summary>
/// The three workflows, in the order they are meant to be run:
///   RunBaseline : one quick training run + reference models (sanity check)
///   RunTuning   : grid search and momentum/weight-decay sweep, VALIDATION data only
///   RunFinal    : retrain the winner and report on the TEST set (once)
/// </summary>
internal static class Experiments
{
    private const string ResultsDir = "results";
    private const int DataSplitSeed = 42;

    // Search space (5 x 5 x 4 = 100 configurations, each repeated over every seed)
    private static readonly double[] LearnRates = { 0.001, 0.005, 0.01, 0.05, 0.1 };
    private static readonly int[] HiddenCounts = { 2, 4, 8, 16, 32 };
    private static readonly int[] EpochCheckpoints = { 100, 500, 1000, 2000 };
    private const double GridMomentum = 0.5;
    private const double GridWeightDecay = 0.0;

    // Secondary sweep around the grid winner
    private static readonly double[] SweepMomentums = { 0.0, 0.5, 0.9 };
    private static readonly double[] SweepWeightDecays = { 0.0, 1e-4, 1e-3 };

    // Independent weight-initialisation / shuffling seeds. Results are reported as mean +/- std.
    private static readonly int[] ModelSeeds = { 1, 2, 3, 4, 5 };

    private static string ResultPath(string fileName) => Path.Combine(ResultsDir, fileName);

    // ---------------------------------------------------------------- baseline

    public static void RunBaseline(string csvPath)
    {
        Console.WriteLine("=== BASELINE RUN ===\n");
        PreparedData data = PreparedData.Load(csvPath, DataSplitSeed);
        data.PrintSummary();

        Console.WriteLine("\nReference models on the validation set:");
        foreach (var (name, metrics) in ReferenceModels(data, data.ValidationScaled))
            Console.WriteLine($"  {name,-26}: {metrics}");

        const int hiddenNodes = 16, epochs = 500;
        const double learnRate = 0.01, momentum = 0.5, weightDecay = 0.0;

        var network = new NeuralNetwork(data.NumInput, hiddenNodes, 1, ModelSeeds[0]);
        Console.WriteLine($"\nNetwork {data.NumInput}-{hiddenNodes}-1 ({network.WeightCount} weights and biases)");
        Console.WriteLine($"epochs={epochs}  learnRate={learnRate}  momentum={momentum}  weightDecay={weightDecay}\n");

        List<EpochRecord> history = network.Train(data.TrainScaled, data.ValidationScaled,
            epochs, learnRate, momentum, weightDecay, logInterval: 50, log: Console.Out);

        ResultsIo.WriteTrainingLog(ResultPath("training_log_baseline.csv"), history, data.Scaler);

        Console.WriteLine("\nNeural network (original units):");
        Console.WriteLine($"  Train      : {ModelEvaluator.Evaluate(data.TrainScaled, network.PredictSingleOutput(data.TrainScaled), data.Scaler)}");
        Console.WriteLine($"  Validation : {ModelEvaluator.Evaluate(data.ValidationScaled, network.PredictSingleOutput(data.ValidationScaled), data.Scaler)}");
    }

    // ------------------------------------------------------------------ tuning

    public static void RunTuning(string csvPath)
    {
        Console.WriteLine("=== HYPERPARAMETER TUNING (validation set only) ===\n");
        PreparedData data = PreparedData.Load(csvPath, DataSplitSeed);
        data.PrintSummary();

        var tuner = new HyperparameterTuner(data, ModelSeeds);

        // 1. Full grid
        Console.WriteLine($"\n[1/2] Grid: {LearnRates.Length} learning rates x {HiddenCounts.Length} hidden sizes x " +
                          $"{EpochCheckpoints.Length} epoch counts, {ModelSeeds.Length} seeds each");
        List<TuningResult> grid = tuner.RunGrid(LearnRates, HiddenCounts, EpochCheckpoints,
            GridMomentum, GridWeightDecay);
        ResultsIo.WriteTuningResults(ResultPath("tuning_results.csv"), grid);

        TuningResult gridBest = TuningResult.Best(grid);
        Console.WriteLine("\nTop 10 grid configurations by mean validation RMSE:");
        PrintTable(grid.OrderBy(r => r.RankingScore).Take(10));

        // 2. Momentum / weight-decay sweep around the grid winner
        Console.WriteLine($"\n[2/2] Sweep around lr={gridBest.Parameters.LearnRate}, " +
                          $"hidden={gridBest.Parameters.HiddenNodes}, epochs={gridBest.Parameters.Epochs}");
        List<TuningResult> sweep = tuner.RunSweep(gridBest.Parameters, SweepMomentums, SweepWeightDecays);
        ResultsIo.WriteTuningResults(ResultPath("sweep_results.csv"), sweep);
        Console.WriteLine("\nSweep results:");
        PrintTable(sweep.OrderBy(r => r.RankingScore));

        TuningResult overallBest = TuningResult.Best(sweep);
        ResultsIo.WriteBestConfig(ResultPath("best_config.csv"), overallBest.Parameters);
        Console.WriteLine($"\nSelected configuration: {overallBest.Parameters}");
        Console.WriteLine($"  validation RMSE {overallBest.ValidationRmseMean:F3} +/- {overallBest.ValidationRmseStd:F3}, " +
                          $"R2 {overallBest.ValidationR2Mean:F4}");

        // 3. Reference models, for context
        Console.WriteLine("\nReference models on the validation set:");
        foreach (var (name, metrics) in ReferenceModels(data, data.ValidationScaled))
            Console.WriteLine($"  {name,-26}: {metrics}");

        // 4. Charts
        Plots.TuningHeatmap(ResultPath("tuning_heatmap.png"), grid, gridBest.Parameters.Epochs,
            LearnRates, HiddenCounts, worstRmse: data.Scaler.TargetStdDev);
        Plots.EpochsChart(ResultPath("epochs_effect.png"), grid, overallBest.Parameters);
    }

    // ------------------------------------------------------------------- final

    private sealed record FinalRun(NeuralNetwork Network, List<EpochRecord> History, RegressionMetrics Test);

    public static void RunFinal(string csvPath)
    {
        Console.WriteLine("=== FINAL EVALUATION (test set, used once) ===\n");
        HyperParameters best = ResultsIo.ReadBestConfig(ResultPath("best_config.csv"));
        PreparedData data = PreparedData.Load(csvPath, DataSplitSeed);
        data.PrintSummary();
        Console.WriteLine($"\nConfiguration: {best}\n");

        // Retrain the chosen configuration with every seed (training set only).
        var runs = new List<FinalRun>();
        for (int i = 0; i < ModelSeeds.Length; ++i)
        {
            Console.WriteLine($"-- seed {ModelSeeds[i]} --");
            var network = new NeuralNetwork(data.NumInput, best.HiddenNodes, 1, ModelSeeds[i]);
            List<EpochRecord> history = network.Train(data.TrainScaled, data.ValidationScaled,
                best.Epochs, best.LearnRate, best.Momentum, best.WeightDecay,
                logInterval: 100, log: Console.Out);

            RegressionMetrics test = ModelEvaluator.Evaluate(
                data.TestScaled, network.PredictSingleOutput(data.TestScaled), data.Scaler);
            runs.Add(new FinalRun(network, history, test));
        }

        // Summary over seeds
        Console.WriteLine("\nTest metrics per seed:");
        for (int i = 0; i < runs.Count; ++i)
            Console.WriteLine($"  seed {ModelSeeds[i]}: {runs[i].Test}");

        var (rmseMean, rmseStd) = Stats.MeanStd(runs.Select(r => r.Test.Rmse).ToArray());
        var (maeMean, _) = Stats.MeanStd(runs.Select(r => r.Test.Mae).ToArray());
        var (r2Mean, _) = Stats.MeanStd(runs.Select(r => r.Test.R2).ToArray());

        var summary = new List<(string Model, double RmseMean, double RmseStd, double MaeMean, double R2Mean)>
        {
            ("Neural network (mean of seeds)", rmseMean, rmseStd, maeMean, r2Mean)
        };
        foreach (var (name, metrics) in ReferenceModels(data, data.TestScaled))
            summary.Add((name, metrics.Rmse, 0.0, metrics.Mae, metrics.R2));

        Console.WriteLine("\nTest-set comparison (original units):");
        Console.WriteLine($"  {"Model",-32} {"RMSE",9} {"MAE",9} {"R2",9}");
        foreach (var row in summary)
        {
            string rmse = row.RmseStd > 0 ? $"{row.RmseMean:F3}+/-{row.RmseStd:F3}" : $"{row.RmseMean:F3}";
            Console.WriteLine($"  {row.Model,-32} {rmse,9} {row.MaeMean,9:F3} {row.R2Mean,9:F4}");
        }
        ResultsIo.WriteFinalSummary(ResultPath("final_summary.csv"), summary);

        // Representative model for the plots: chosen by VALIDATION loss, never by test loss.
        FinalRun representative = runs.MinBy(r => r.History[^1].ValidationMse)!;
        ResultsIo.WriteTrainingLog(ResultPath("training_log.csv"), representative.History, data.Scaler);

        var (actual, predicted) = ModelEvaluator.ToOriginalUnits(
            data.TestScaled, representative.Network.PredictSingleOutput(data.TestScaled), data.Scaler);
        ResultsIo.WritePredictions(ResultPath("predictions_test.csv"), actual, predicted);

        Plots.LossCurves(ResultPath("loss_curves.png"), representative.History, data.Scaler, data.TargetName);
        Plots.PredictedVsActual(ResultPath("predicted_vs_actual.png"), actual, predicted, data.TargetName);
        Plots.Residuals(ResultPath("residuals.png"), actual, predicted, data.TargetName);
    }

    // ----------------------------------------------------------------- helpers

    /// <summary>Predict-the-training-mean and ordinary linear regression, both fitted on training rows.</summary>
    private static List<(string Name, RegressionMetrics Metrics)> ReferenceModels(
        PreparedData data, double[][] scaledRows)
    {
        // In scaled space the training mean is 0, so "predict the mean" means predicting 0.
        RegressionMetrics meanModel = ModelEvaluator.Evaluate(
            scaledRows, new double[scaledRows.Length], data.Scaler);

        var linear = new LinearRegressionBaseline();
        linear.Fit(data.TrainScaled);
        RegressionMetrics linearModel = ModelEvaluator.Evaluate(
            scaledRows, linear.Predict(scaledRows), data.Scaler);

        return new List<(string, RegressionMetrics)>
        {
            ("Predict training mean", meanModel),
            ("Linear regression (OLS)", linearModel)
        };
    }

    private static void PrintTable(IEnumerable<TuningResult> results)
    {
        Console.WriteLine($"  {"lr",7} {"hidden",6} {"epochs",6} {"mom",5} {"decay",7} " +
                          $"{"train RMSE",16} {"val RMSE",16} {"val R2",7} {"gap",7}");
        foreach (TuningResult r in results)
        {
            HyperParameters p = r.Parameters;
            Console.WriteLine($"  {p.LearnRate,7} {p.HiddenNodes,6} {p.Epochs,6} {p.Momentum,5} {p.WeightDecay,7} " +
                              $"{r.TrainRmseMean,8:F3}+/-{r.TrainRmseStd,5:F3} " +
                              $"{r.ValidationRmseMean,8:F3}+/-{r.ValidationRmseStd,5:F3} " +
                              $"{r.ValidationR2Mean,7:F4} {r.OverfitGap,7:F3}");
        }
    }
}