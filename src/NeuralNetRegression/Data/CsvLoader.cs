using System.Globalization;
using Microsoft.VisualBasic.FileIO;

namespace NeuralNetRegression.Data;

public sealed class CsvDataset
{
    public required string[] Headers { get; init; }
    public required double[][] Rows { get; init; }
    public int DroppedRows { get; init; }
}

public static class CsvLoader
{
    private static readonly HashSet<string> MissingTokens =
        new(StringComparer.OrdinalIgnoreCase)
        { "", "?", "NA", "N/A", "NaN", "null" };

    public static CsvDataset Load(string path, char delimiter = ',')
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"CSV file not found: {path}");

        using var parser = new TextFieldParser(path)
        {
            TextFieldType = FieldType.Delimited,
            HasFieldsEnclosedInQuotes = true,
            TrimWhiteSpace = true
        };
        parser.SetDelimiters(delimiter.ToString());

        string[] headers = parser.ReadFields()
            ?? throw new InvalidDataException("CSV header is missing.");

        var rows = new List<double[]>();
        int dropped = 0;

        while (!parser.EndOfData)
        {
            string[]? fields;
            try { fields = parser.ReadFields(); }
            catch (MalformedLineException) { dropped++; continue; }

            if (fields is null || fields.Length < headers.Length)
            {
                dropped++;
                continue;
            }

            var row = new double[headers.Length];
            bool valid = true;

            for (int i = 0; i < headers.Length; i++)
            {
                string value = fields[i].Trim();

                if (MissingTokens.Contains(value) ||
                    !double.TryParse(value, NumberStyles.Float,
                        CultureInfo.InvariantCulture, out row[i]))
                {
                    valid = false;
                    break;
                }
            }

            if (valid) rows.Add(row);
            else dropped++;
        }

        if (rows.Count == 0)
            throw new InvalidDataException("No valid numeric rows were found.");

        return new CsvDataset
        {
            Headers = headers,
            Rows = rows.ToArray(),
            DroppedRows = dropped
        };
    }
}