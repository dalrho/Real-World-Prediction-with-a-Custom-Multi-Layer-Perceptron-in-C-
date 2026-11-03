namespace NeuralNetRegression;

/// <summary>
/// Command-line entry point. Run from the repository root:
///   dotnet run --project src/NeuralNetRegression -- baseline [csv]
///   dotnet run --project src/NeuralNetRegression -- tune     [csv]
///   dotnet run --project src/NeuralNetRegression -- final    [csv]
///   dotnet run --project src/NeuralNetRegression -- all      [csv]   (tune, then final)
/// The CSV path defaults to data/concrete.csv; the target must be the last column.
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        string command = args.Length > 0 ? args[0].ToLowerInvariant() : "help";
        string csvPath = args.Length > 1 ? args[1] : Path.Combine("data", "concrete.csv");

        try
        {
            switch (command)
            {
                case "baseline":
                    Experiments.RunBaseline(csvPath);
                    break;
                case "tune":
                    Experiments.RunTuning(csvPath);
                    break;
                case "final":
                    Experiments.RunFinal(csvPath);
                    break;
                case "all":
                    Experiments.RunTuning(csvPath);
                    Console.WriteLine();
                    Experiments.RunFinal(csvPath);
                    break;
                default:
                    PrintUsage();
                    return command == "help" ? 0 : 1;
            }
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }

        return 0;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("Usage: dotnet run --project src/NeuralNetRegression -- <command> [csvPath]");
        Console.WriteLine("  baseline  quick sanity-check training run + reference models");
        Console.WriteLine("  tune      grid search + momentum/weight-decay sweep (validation set only)");
        Console.WriteLine("  final     retrain the chosen configuration and evaluate on the test set");
        Console.WriteLine("  all       tune, then final");
    }
}