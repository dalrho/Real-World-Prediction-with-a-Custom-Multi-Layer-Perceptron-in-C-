using NeuralNetRegression.Data;
using NeuralNetRegression.Tuning;
using ScottPlot;

namespace NeuralNetRegression.Plotting;

/// <summary>
/// All charts, saved as PNG files via ScottPlot 5 (works on Linux).
/// If a ScottPlot method name differs in your installed 5.x version, only this file needs adjusting.
/// </summary>
public static class Plots
{
    /// <summary>Training vs validation RMSE per epoch (the "loss decreasing" figure).</summary>
    public static void LossCurves(string path, List<EpochRecord> history, StandardScaler scaler, string unit)
    {
        double[] epochs = history.Select(h => (double)h.Epoch).ToArray();
        double[] train = history.Select(h => Math.Sqrt(h.TrainMse) * scaler.TargetStdDev).ToArray();
        double[] validation = history.Select(h => Math.Sqrt(h.ValidationMse) * scaler.TargetStdDev).ToArray();

        var plot = new Plot();
        var trainLine = plot.Add.Scatter(epochs, train);
        trainLine.LegendText = "Training RMSE";
        trainLine.MarkerSize = 0;
        var validationLine = plot.Add.Scatter(epochs, validation);
        validationLine.LegendText = "Validation RMSE";
        validationLine.MarkerSize = 0;

        plot.Title("Loss during training");
        plot.XLabel("Epoch");
        plot.YLabel($"RMSE ({unit})");
        plot.ShowLegend();
        Save(plot, path, 900, 550);
    }

    /// <summary>Predicted vs actual with the ideal y = x diagonal.</summary>
    public static void PredictedVsActual(string path, double[] actual, double[] predicted, string target)
    {
        var plot = new Plot();
        var points = plot.Add.ScatterPoints(actual, predicted);
        points.LegendText = "Test samples";
        points.MarkerSize = 6;

        double low = Math.Min(actual.Min(), predicted.Min());
        double high = Math.Max(actual.Max(), predicted.Max());
        var diagonal = plot.Add.Line(low, low, high, high);
        diagonal.LegendText = "Perfect prediction (y = x)";
        diagonal.LineWidth = 2;

        plot.Title("Predicted vs actual (test set)");
        plot.XLabel($"Actual {target}");
        plot.YLabel($"Predicted {target}");
        plot.ShowLegend();
        Save(plot, path, 700, 700);
    }

    /// <summary>Residuals (actual - predicted) against predicted values.</summary>
    public static void Residuals(string path, double[] actual, double[] predicted, string target)
    {
        double[] residuals = actual.Zip(predicted, (a, p) => a - p).ToArray();

        var plot = new Plot();
        var points = plot.Add.ScatterPoints(predicted, residuals);
        points.MarkerSize = 6;
        var zero = plot.Add.Line(predicted.Min(), 0, predicted.Max(), 0);
        zero.LineWidth = 2;

        plot.Title("Residuals (test set)");
        plot.XLabel($"Predicted {target}");
        plot.YLabel("Residual (actual - predicted)");
        Save(plot, path, 900, 550);
    }

    /// <summary>
    /// Heatmap of mean validation RMSE: learning rate (x) by hidden nodes (y), for one epoch count.
    /// Cells are drawn as coloured rectangles so row/column order is fully under our control.
    /// Green = low error, red = error at or above 'worstRmse' (diverged runs are labelled "div.").
    /// </summary>
    public static void TuningHeatmap(string path, IReadOnlyList<TuningResult> grid, int epochs,
        double[] learnRates, int[] hiddenCounts, double worstRmse)
    {
        var plot = new Plot();

        double[] finite = grid
            .Where(r => r.Parameters.Epochs == epochs && !double.IsNaN(r.ValidationRmseMean))
            .Select(r => r.ValidationRmseMean).ToArray();
        double best = finite.Length > 0 ? finite.Min() : 0.0;

        for (int row = 0; row < hiddenCounts.Length; ++row)
        {
            for (int col = 0; col < learnRates.Length; ++col)
            {
                TuningResult? cell = grid.FirstOrDefault(r =>
                    r.Parameters.Epochs == epochs &&
                    r.Parameters.HiddenNodes == hiddenCounts[row] &&
                    r.Parameters.LearnRate == learnRates[col]);
                if (cell is null) continue;

                bool diverged = double.IsNaN(cell.ValidationRmseMean);
                double fraction = diverged ? 1.0 : (cell.ValidationRmseMean - best) / Math.Max(worstRmse - best, 1e-9);

                var rectangle = plot.Add.Rectangle(col, col + 1, row, row + 1);
                rectangle.FillColor = ScoreColor(fraction);
                rectangle.LineColor = Colors.White;
                rectangle.LineWidth = 2;

                var label = plot.Add.Text(diverged ? "div." : cell.ValidationRmseMean.ToString("F2"),
                    col + 0.5, row + 0.5);
                label.Alignment = Alignment.MiddleCenter;
            }
        }

        Tick[] xTicks = learnRates.Select((lr, i) => new Tick(i + 0.5, lr.ToString("G"))).ToArray();
        Tick[] yTicks = hiddenCounts.Select((h, i) => new Tick(i + 0.5, h.ToString())).ToArray();
        plot.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(xTicks);
        plot.Axes.Left.TickGenerator = new ScottPlot.TickGenerators.NumericManual(yTicks);
        plot.Axes.SetLimits(0, learnRates.Length, 0, hiddenCounts.Length);

        plot.Title($"Validation RMSE at {epochs} epochs (mean over seeds)");
        plot.XLabel("Learning rate");
        plot.YLabel("Hidden nodes");
        Save(plot, path, 800, 600);
    }

    /// <summary>Train vs validation RMSE at each epoch checkpoint for the winning (learn rate, hidden) pair.</summary>
    public static void EpochsChart(string path, IReadOnlyList<TuningResult> grid, HyperParameters best)
    {
        List<TuningResult> series = grid
            .Where(r => r.Parameters.LearnRate == best.LearnRate && r.Parameters.HiddenNodes == best.HiddenNodes)
            .OrderBy(r => r.Parameters.Epochs)
            .ToList();

        double[] epochs = series.Select(r => (double)r.Parameters.Epochs).ToArray();

        var plot = new Plot();
        var train = plot.Add.Scatter(epochs, series.Select(r => r.TrainRmseMean).ToArray());
        train.LegendText = "Training RMSE";
        var validation = plot.Add.Scatter(epochs, series.Select(r => r.ValidationRmseMean).ToArray());
        validation.LegendText = "Validation RMSE";

        plot.Title($"Effect of epochs (lr = {best.LearnRate}, hidden = {best.HiddenNodes})");
        plot.XLabel("Epochs");
        plot.YLabel("RMSE (mean over seeds)");
        plot.ShowLegend();
        Save(plot, path, 800, 550);
    }

    // ---------------------------------------------------------------- helpers

    private static void Save(Plot plot, string path, int width, int height)
    {
        string? directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
        plot.SavePng(path, width, height);
        Console.WriteLine($"Saved plot: {path}");
    }

    /// <summary>0 = best (green) -> 0.5 = yellow -> 1 = worst (red). Light shades keep black labels readable.</summary>
    private static Color ScoreColor(double fraction)
    {
        fraction = Math.Clamp(fraction, 0.0, 1.0);
        (byte R, byte G, byte B) green = (120, 200, 130);
        (byte R, byte G, byte B) yellow = (250, 230, 130);
        (byte R, byte G, byte B) red = (235, 110, 110);

        var (from, to, t) = fraction < 0.5
            ? (green, yellow, fraction / 0.5)
            : (yellow, red, (fraction - 0.5) / 0.5);

        return new Color(Lerp(from.R, to.R, t), Lerp(from.G, to.G, t), Lerp(from.B, to.B, t));
    }

    private static byte Lerp(byte a, byte b, double t) => (byte)Math.Round(a + (b - a) * t);
}