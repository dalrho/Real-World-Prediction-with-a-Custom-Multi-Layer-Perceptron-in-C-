namespace NeuralNetRegression.Data;

public sealed class StandardScaler
{
    private double[] means = Array.Empty<double>();
    private double[] stdDevs = Array.Empty<double>();

    public bool IsFitted => means.Length > 0;

    public double TargetMean => means[^1];

    public double TargetStdDev => stdDevs[^1];

    public void Fit(double[][] trainingRows)
    {
        int columns = trainingRows[0].Length;
        means = new double[columns];
        stdDevs = new double[columns];

        for (int c = 0; c < columns; ++c)
        {
            double sum = 0.0;
            foreach (double[] row in trainingRows) sum += row[c];
            means[c] = sum / trainingRows.Length;

            double sumSq = 0.0;
            foreach (double[] row in trainingRows)
            {
                double diff = row[c] - means[c];
                sumSq += diff * diff;
            }
            double std = Math.Sqrt(sumSq / trainingRows.Length);

            stdDevs[c] = std > 1e-12 ? std : 1.0;
        }
    }

    public double[][] Transform(double[][] rows)
    {
        if (!IsFitted) throw new InvalidOperationException("Call Fit() before Transform().");

        var result = new double[rows.Length][];
        for (int r = 0; r < rows.Length; ++r)
        {
            result[r] = new double[rows[r].Length];
            for (int c = 0; c < rows[r].Length; ++c)
                result[r][c] = (rows[r][c] - means[c]) / stdDevs[c];
        }
        return result;
    }

    public double InverseTarget(double scaledTarget) =>
        scaledTarget * TargetStdDev + TargetMean;
}