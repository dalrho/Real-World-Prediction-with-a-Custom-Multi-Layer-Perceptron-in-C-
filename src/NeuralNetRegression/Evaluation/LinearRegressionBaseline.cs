namespace NeuralNetRegression.Evaluation;

/// <summary>
/// Ordinary least squares with an intercept, solved from the normal equations (X'X) w = X'y
/// by Gauss-Jordan elimination. Used only as a reference point for the neural network.
/// Rows are laid out as [x_0 ... x_(p-1), target].
/// </summary>
public sealed class LinearRegressionBaseline
{
    private double[] weights = Array.Empty<double>();   // p coefficients followed by the intercept

    public void Fit(double[][] rows)
    {
        int featureCount = rows[0].Length - 1;
        int size = featureCount + 1;   // +1 for the intercept

        // Augmented matrix [X'X | X'y]
        var a = new double[size][];
        for (int i = 0; i < size; ++i)
            a[i] = new double[size + 1];

        foreach (double[] row in rows)
        {
            double target = row[featureCount];
            for (int i = 0; i < size; ++i)
            {
                double xi = i < featureCount ? row[i] : 1.0;
                for (int j = 0; j < size; ++j)
                {
                    double xj = j < featureCount ? row[j] : 1.0;
                    a[i][j] += xi * xj;
                }
                a[i][size] += xi * target;
            }
        }

        for (int i = 0; i < size; ++i)
            a[i][i] += 1e-8;   // tiny ridge keeps the system solvable if features are collinear

        for (int col = 0; col < size; ++col)
        {
            int pivot = col;
            for (int r = col + 1; r < size; ++r)
                if (Math.Abs(a[r][col]) > Math.Abs(a[pivot][col])) pivot = r;
            (a[col], a[pivot]) = (a[pivot], a[col]);

            double pivotValue = a[col][col];
            for (int c = col; c <= size; ++c)
                a[col][c] /= pivotValue;

            for (int r = 0; r < size; ++r)
            {
                if (r == col) continue;
                double factor = a[r][col];
                for (int c = col; c <= size; ++c)
                    a[r][c] -= factor * a[col][c];
            }
        }

        weights = new double[size];
        for (int i = 0; i < size; ++i)
            weights[i] = a[i][size];
    }

    public double[] Predict(double[][] rows)
    {
        int featureCount = weights.Length - 1;
        var predictions = new double[rows.Length];
        for (int r = 0; r < rows.Length; ++r)
        {
            double sum = weights[featureCount];
            for (int i = 0; i < featureCount; ++i)
                sum += weights[i] * rows[r][i];
            predictions[r] = sum;
        }
        return predictions;
    }
}