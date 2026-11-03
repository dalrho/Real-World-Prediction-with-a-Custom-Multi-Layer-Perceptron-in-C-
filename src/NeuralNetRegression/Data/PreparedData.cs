namespace NeuralNetRegression.Data;

/// <summary>
/// One-stop container for a ready-to-train dataset:
/// CSV -> 70/15/15 split -> z-score scaling fitted on the training rows only.
/// </summary>
public sealed class PreparedData
{
    public required string[] FeatureNames { get; init; }
    public required string TargetName { get; init; }
    public required StandardScaler Scaler { get; init; }

    public required double[][] TrainScaled { get; init; }
    public required double[][] ValidationScaled { get; init; }
    public required double[][] TestScaled { get; init; }

    public int TotalRows { get; init; }
    public int DroppedRows { get; init; }

    public int NumInput => FeatureNames.Length;

    public static PreparedData Load(string csvPath, int splitSeed,
        double trainFraction = 0.70, double validationFraction = 0.15)
    {
        CsvDataset dataset = CsvLoader.Load(csvPath);
        DataSplit split = DataSplitter.Split(dataset.Rows, trainFraction, validationFraction, splitSeed);

        var scaler = new StandardScaler();
        scaler.Fit(split.Train);   // statistics come from the training rows ONLY

        return new PreparedData
        {
            FeatureNames = dataset.Headers[..^1],
            TargetName = dataset.Headers[^1],
            Scaler = scaler,
            TrainScaled = scaler.Transform(split.Train),
            ValidationScaled = scaler.Transform(split.Validation),
            TestScaled = scaler.Transform(split.Test),
            TotalRows = dataset.Rows.Length,
            DroppedRows = dataset.DroppedRows
        };
    }

    public void PrintSummary()
    {
        Console.WriteLine($"Loaded {TotalRows} rows ({DroppedRows} dropped for missing/invalid values)");
        Console.WriteLine($"Features ({NumInput}): {string.Join(", ", FeatureNames)}");
        Console.WriteLine($"Target: {TargetName}  (mean {Scaler.TargetMean:F2}, std {Scaler.TargetStdDev:F2} on training rows)");
        Console.WriteLine($"Split -> train {TrainScaled.Length}, validation {ValidationScaled.Length}, test {TestScaled.Length}");
    }
}