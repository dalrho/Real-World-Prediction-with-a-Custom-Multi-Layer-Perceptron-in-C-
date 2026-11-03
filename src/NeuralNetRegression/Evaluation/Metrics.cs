namespace NeuralNetRegression;

public sealed record RegressionMetrics(double Mse, double Rmse, double Mae, double R2)
{
    public override string ToString() =>
        $"MSE={Mse:F3}  RMSE={Rmse:F3}  MAE={Mae:F3}  R2={R2:F4}";
}

public static class Metrics
{
    public static RegressionMetrics Compute(double[] actual, double[] predicted)
    {
        if (actual.Length != predicted.Length || actual.Length == 0)
            throw new ArgumentException("Actual and predicted must be the same non-zero length.");

        int n = actual.Length;
        double meanActual = actual.Average();

        double sumSquaredError = 0.0;
        double sumAbsError = 0.0;
        double totalSumOfSquares = 0.0;

        for (int i = 0; i < n; ++i)
        {
            double error = actual[i] - predicted[i];
            sumSquaredError += error * error;
            sumAbsError += Math.Abs(error);

            double deviation = actual[i] - meanActual;
            totalSumOfSquares += deviation * deviation;
        }

        double mse = sumSquaredError / n;
        double r2 = totalSumOfSquares > 0.0 ? 1.0 - sumSquaredError / totalSumOfSquares : 0.0;

        return new RegressionMetrics(mse, Math.Sqrt(mse), sumAbsError / n, r2);
    }
}