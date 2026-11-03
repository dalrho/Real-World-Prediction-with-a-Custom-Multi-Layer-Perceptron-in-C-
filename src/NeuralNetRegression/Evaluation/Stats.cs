namespace NeuralNetRegression.Evaluation;

public static class Stats
{
    /// <summary>Mean and sample standard deviation (n - 1). NaN inputs propagate to NaN outputs.</summary>
    public static (double Mean, double StdDev) MeanStd(IReadOnlyList<double> values)
    {
        double mean = values.Average();
        if (values.Count < 2) return (mean, 0.0);

        double sumSquares = values.Sum(v => (v - mean) * (v - mean));
        return (mean, Math.Sqrt(sumSquares / (values.Count - 1)));
    }
}