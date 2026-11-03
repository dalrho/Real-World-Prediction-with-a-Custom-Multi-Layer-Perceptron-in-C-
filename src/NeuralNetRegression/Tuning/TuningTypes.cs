namespace NeuralNetRegression.Tuning;

/// <summary>One complete hyperparameter configuration.</summary>
public sealed record HyperParameters(
    double LearnRate, int HiddenNodes, int Epochs, double Momentum, double WeightDecay);

/// <summary>
/// Aggregated outcome of one configuration over several random seeds.
/// RMSE values are in the target's original units (e.g. MPa).
/// </summary>
public sealed record TuningResult(
    HyperParameters Parameters,
    int SeedCount,
    double TrainRmseMean,
    double TrainRmseStd,
    double ValidationRmseMean,
    double ValidationRmseStd,
    double ValidationR2Mean)
{
    /// <summary>Validation minus training RMSE. A large positive gap signals overfitting.</summary>
    public double OverfitGap => ValidationRmseMean - TrainRmseMean;

    /// <summary>Selection score: lower is better; diverged runs (NaN) rank last.</summary>
    public double RankingScore =>
        double.IsNaN(ValidationRmseMean) ? double.PositiveInfinity : ValidationRmseMean;

    public static TuningResult Best(IEnumerable<TuningResult> results) =>
        results.MinBy(r => r.RankingScore)
        ?? throw new InvalidOperationException("No tuning results to choose from.");
}