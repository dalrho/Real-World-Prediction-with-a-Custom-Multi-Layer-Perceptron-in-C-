namespace NeuralNetRegression.Data;

public sealed record DataSplit(double[][] Train, double[][] Validation, double[][] Test);

public static class DataSplitter
{
    public static DataSplit Split(double[][] data, double trainFraction,
        double validationFraction, int seed)
    {
        if (trainFraction <= 0 || validationFraction < 0 || trainFraction + validationFraction >= 1.0)
            throw new ArgumentException("Fractions must leave a non-empty test set.");

        double[][] shuffled = (double[][])data.Clone();
        var rnd = new Random(seed);
        for (int i = shuffled.Length - 1; i > 0; --i)
        {
            int j = rnd.Next(i + 1);
            (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
        }

        int trainCount = (int)(shuffled.Length * trainFraction);
        int validationCount = (int)(shuffled.Length * validationFraction);

        return new DataSplit(
            Train: shuffled[..trainCount],
            Validation: shuffled[trainCount..(trainCount + validationCount)],
            Test: shuffled[(trainCount + validationCount)..]);
    }
}