using NeuralNetRegression.Data;
using NeuralNetRegression.Evaluation;

namespace NeuralNetRegression.Tuning;

/// <summary>
/// Systematic hyperparameter search. Models are fitted on the TRAINING set and scored on the
/// VALIDATION set; the test set is never touched here.
///
/// Efficiency trick: training is deterministic for a given seed and has no epoch-dependent
/// schedule, so a model trained for 2000 epochs passes through exactly the same states as models
/// trained for 100, 500 and 1000 epochs. One run is therefore read at several epoch checkpoints.
/// Runs are independent (each owns its network and RNG), so they execute in parallel.
/// </summary>
public sealed class HyperparameterTuner
{
    private sealed record Config(double LearnRate, int HiddenNodes, double Momentum, double WeightDecay);

    private readonly PreparedData data;
    private readonly int[] seeds;
    private readonly double validationTargetVariance;   // variance of scaled validation targets

    public HyperparameterTuner(PreparedData data, int[] seeds)
    {
        this.data = data;
        this.seeds = seeds;

        double[] targets = data.ValidationScaled.Select(row => row[^1]).ToArray();
        double mean = targets.Average();
        validationTargetVariance = targets.Sum(t => (t - mean) * (t - mean)) / targets.Length;
    }

    /// <summary>Full grid over learning rate x hidden nodes x epoch checkpoints.</summary>
    public List<TuningResult> RunGrid(double[] learnRates, int[] hiddenCounts, int[] epochCheckpoints,
        double momentum, double weightDecay)
    {
        var configs = new List<Config>();
        foreach (double learnRate in learnRates)
            foreach (int hidden in hiddenCounts)
                configs.Add(new Config(learnRate, hidden, momentum, weightDecay));

        return Evaluate(configs, epochCheckpoints);
    }

    /// <summary>Momentum x weight-decay sweep around an already chosen (learn rate, hidden, epochs).</summary>
    public List<TuningResult> RunSweep(HyperParameters best, double[] momentums, double[] weightDecays)
    {
        var configs = new List<Config>();
        foreach (double momentum in momentums)
            foreach (double decay in weightDecays)
                configs.Add(new Config(best.LearnRate, best.HiddenNodes, momentum, decay));

        return Evaluate(configs, new[] { best.Epochs });
    }

    private List<TuningResult> Evaluate(List<Config> configs, int[] checkpoints)
    {
        int maxEpochs = checkpoints.Max();
        int totalRuns = configs.Count * seeds.Length;
        double targetStd = data.Scaler.TargetStdDev;

        // [config][seed][checkpoint]; every parallel job writes only its own slot.
        double[][][] trainRmse = NewCube(configs.Count, seeds.Length, checkpoints.Length);
        double[][][] validationRmse = NewCube(configs.Count, seeds.Length, checkpoints.Length);
        double[][][] validationR2 = NewCube(configs.Count, seeds.Length, checkpoints.Length);

        int finished = 0;
        Parallel.For(0, totalRuns, runIndex =>
        {
            int configIndex = runIndex / seeds.Length;
            int seedIndex = runIndex % seeds.Length;
            Config config = configs[configIndex];

            var network = new NeuralNetwork(data.NumInput, config.HiddenNodes, 1, seeds[seedIndex]);
            List<EpochRecord> history = network.Train(
                data.TrainScaled, data.ValidationScaled, maxEpochs,
                config.LearnRate, config.Momentum, config.WeightDecay);

            for (int k = 0; k < checkpoints.Length; ++k)
            {
                EpochRecord record = history[checkpoints[k] - 1];
                trainRmse[configIndex][seedIndex][k] = Math.Sqrt(record.TrainMse) * targetStd;
                validationRmse[configIndex][seedIndex][k] = Math.Sqrt(record.ValidationMse) * targetStd;
                // R2 = 1 - MSE / Var(target); the scaling cancels, so scaled values are fine.
                validationR2[configIndex][seedIndex][k] = 1.0 - record.ValidationMse / validationTargetVariance;
            }

            int done = Interlocked.Increment(ref finished);
            if (done % 10 == 0 || done == totalRuns)
                Console.WriteLine($"  {done}/{totalRuns} runs complete");
        });

        var results = new List<TuningResult>();
        for (int c = 0; c < configs.Count; ++c)
        {
            for (int k = 0; k < checkpoints.Length; ++k)
            {
                double[] train = seeds.Select((_, s) => trainRmse[c][s][k]).ToArray();
                double[] validation = seeds.Select((_, s) => validationRmse[c][s][k]).ToArray();
                double[] r2 = seeds.Select((_, s) => validationR2[c][s][k]).ToArray();

                var (trainMean, trainStd) = Stats.MeanStd(train);
                var (validationMean, validationStd) = Stats.MeanStd(validation);

                results.Add(new TuningResult(
                    new HyperParameters(configs[c].LearnRate, configs[c].HiddenNodes, checkpoints[k],
                        configs[c].Momentum, configs[c].WeightDecay),
                    seeds.Length, trainMean, trainStd, validationMean, validationStd, r2.Average()));
            }
        }
        return results;
    }

    private static double[][][] NewCube(int a, int b, int c)
    {
        var cube = new double[a][][];
        for (int i = 0; i < a; ++i)
        {
            cube[i] = new double[b][];
            for (int j = 0; j < b; ++j)
                cube[i][j] = new double[c];
        }
        return cube;
    }
}