using System.Globalization;
using NeuralNetRegression.Data;

namespace NeuralNetRegression.Tuning;

/// <summary>Reads and writes every CSV artefact (training log, tuning tables, predictions).</summary>
public static class ResultsIo
{
    private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

    private static string Fmt(double value) =>
        double.IsNaN(value) ? "NaN" : value.ToString("G6", Inv);

    private static StreamWriter Open(string path)
    {
        string? directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
        return new StreamWriter(path);
    }

    /// <summary>Epoch-by-epoch loss. RMSE columns are in the target's original units.</summary>
    public static void WriteTrainingLog(string path, List<EpochRecord> history, StandardScaler scaler)
    {
        using StreamWriter writer = Open(path);
        writer.WriteLine("epoch,train_mse_scaled,validation_mse_scaled,train_rmse_original,validation_rmse_original");
        foreach (EpochRecord r in history)
        {
            writer.WriteLine(string.Join(",",
                r.Epoch.ToString(Inv),
                Fmt(r.TrainMse),
                Fmt(r.ValidationMse),
                Fmt(Math.Sqrt(r.TrainMse) * scaler.TargetStdDev),
                Fmt(Math.Sqrt(r.ValidationMse) * scaler.TargetStdDev)));
        }
    }

    public static void WriteTuningResults(string path, IEnumerable<TuningResult> results)
    {
        using StreamWriter writer = Open(path);
        writer.WriteLine("learn_rate,hidden_nodes,epochs,momentum,weight_decay,seeds," +
                         "train_rmse_mean,train_rmse_std,val_rmse_mean,val_rmse_std,val_r2_mean,overfit_gap");
        foreach (TuningResult r in results)
        {
            HyperParameters p = r.Parameters;
            writer.WriteLine(string.Join(",",
                p.LearnRate.ToString("R", Inv), p.HiddenNodes.ToString(Inv), p.Epochs.ToString(Inv),
                p.Momentum.ToString("R", Inv), p.WeightDecay.ToString("R", Inv), r.SeedCount.ToString(Inv),
                Fmt(r.TrainRmseMean), Fmt(r.TrainRmseStd),
                Fmt(r.ValidationRmseMean), Fmt(r.ValidationRmseStd),
                Fmt(r.ValidationR2Mean), Fmt(r.OverfitGap)));
        }
    }

    public static void WriteBestConfig(string path, HyperParameters p)
    {
        using StreamWriter writer = Open(path);
        writer.WriteLine("learn_rate,hidden_nodes,epochs,momentum,weight_decay");
        writer.WriteLine(string.Join(",",
            p.LearnRate.ToString("R", Inv), p.HiddenNodes.ToString(Inv), p.Epochs.ToString(Inv),
            p.Momentum.ToString("R", Inv), p.WeightDecay.ToString("R", Inv)));
    }

    public static HyperParameters ReadBestConfig(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"{path} not found. Run the 'tune' command first.");

        string[] cells = File.ReadAllLines(path)[1].Split(',');
        return new HyperParameters(
            double.Parse(cells[0], Inv), int.Parse(cells[1], Inv), int.Parse(cells[2], Inv),
            double.Parse(cells[3], Inv), double.Parse(cells[4], Inv));
    }

    public static void WritePredictions(string path, double[] actual, double[] predicted)
    {
        using StreamWriter writer = Open(path);
        writer.WriteLine("actual,predicted,residual");
        for (int i = 0; i < actual.Length; ++i)
            writer.WriteLine($"{Fmt(actual[i])},{Fmt(predicted[i])},{Fmt(actual[i] - predicted[i])}");
    }

    public static void WriteFinalSummary(string path,
        IEnumerable<(string Model, double RmseMean, double RmseStd, double MaeMean, double R2Mean)> rows)
    {
        using StreamWriter writer = Open(path);
        writer.WriteLine("model,test_rmse_mean,test_rmse_std,test_mae_mean,test_r2_mean");
        foreach (var row in rows)
            writer.WriteLine($"{row.Model},{Fmt(row.RmseMean)},{Fmt(row.RmseStd)},{Fmt(row.MaeMean)},{Fmt(row.R2Mean)}");
    }
}