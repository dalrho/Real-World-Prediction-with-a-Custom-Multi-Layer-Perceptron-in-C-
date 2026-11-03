using NeuralNetRegression.Data;

namespace NeuralNetRegression.Evaluation;

/// <summary>Turns scaled predictions back into original units (e.g. MPa) and scores them.</summary>
public static class ModelEvaluator
{
    public static (double[] Actual, double[] Predicted) ToOriginalUnits(
        double[][] scaledRows, double[] scaledPredictions, StandardScaler scaler)
    {
        int targetColumn = scaledRows[0].Length - 1;
        var actual = new double[scaledRows.Length];
        var predicted = new double[scaledRows.Length];

        for (int i = 0; i < scaledRows.Length; ++i)
        {
            actual[i] = scaler.InverseTarget(scaledRows[i][targetColumn]);
            predicted[i] = scaler.InverseTarget(scaledPredictions[i]);
        }
        return (actual, predicted);
    }

    public static RegressionMetrics Evaluate(double[][] scaledRows, double[] scaledPredictions,
        StandardScaler scaler)
    {
        var (actual, predicted) = ToOriginalUnits(scaledRows, scaledPredictions, scaler);
        return Metrics.Compute(actual, predicted);
    }
}