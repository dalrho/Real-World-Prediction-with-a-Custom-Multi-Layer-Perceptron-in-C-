namespace NeuralNetRegression;

public sealed record EpochRecord(int Epoch, double TrainMse, double ValidationMse);

public sealed class NeuralNetwork
{
    private readonly int numInput;
    private readonly int numHidden;
    private readonly int numOutput;

    private readonly double[] inputs;
    private readonly double[][] ihWeights;
    private readonly double[] hBiases;
    private readonly double[] hOutputs;

    private readonly double[][] hoWeights;
    private readonly double[] oBiases;
    private readonly double[] outputs;

    private readonly Random rnd;

    public NeuralNetwork(int numInput, int numHidden, int numOutput, int seed)
    {
        this.numInput = numInput;
        this.numHidden = numHidden;
        this.numOutput = numOutput;

        inputs = new double[numInput];
        ihWeights = MakeMatrix(numInput, numHidden);
        hBiases = new double[numHidden];
        hOutputs = new double[numHidden];

        hoWeights = MakeMatrix(numHidden, numOutput);
        oBiases = new double[numOutput];
        outputs = new double[numOutput];

        rnd = new Random(seed);
        InitializeWeights();
    }

    public int WeightCount =>
        (numInput * numHidden) + numHidden + (numHidden * numOutput) + numOutput;


    private static double[][] MakeMatrix(int rows, int cols)
    {
        var result = new double[rows][];
        for (int r = 0; r < rows; ++r)
            result[r] = new double[cols];
        return result;
    }

    private void InitializeWeights()
    {
        double ihLimit = 1.0 / Math.Sqrt(numInput);
        double hoLimit = 1.0 / Math.Sqrt(numHidden);

        for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
                ihWeights[i][j] = (2.0 * rnd.NextDouble() - 1.0) * ihLimit;

        for (int j = 0; j < numHidden; ++j)
            for (int k = 0; k < numOutput; ++k)
                hoWeights[j][k] = (2.0 * rnd.NextDouble() - 1.0) * hoLimit;
    }

    public double[] GetWeights()
    {
        var result = new double[WeightCount];
        int k = 0;
        for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
                result[k++] = ihWeights[i][j];
        for (int j = 0; j < numHidden; ++j)
            result[k++] = hBiases[j];
        for (int j = 0; j < numHidden; ++j)
            for (int o = 0; o < numOutput; ++o)
                result[k++] = hoWeights[j][o];
        for (int o = 0; o < numOutput; ++o)
            result[k++] = oBiases[o];
        return result;
    }

    public void SetWeights(double[] weights)
    {
        if (weights.Length != WeightCount)
            throw new ArgumentException("Weights array has the wrong length.", nameof(weights));

        int k = 0;
        for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
                ihWeights[i][j] = weights[k++];
        for (int j = 0; j < numHidden; ++j)
            hBiases[j] = weights[k++];
        for (int j = 0; j < numHidden; ++j)
            for (int o = 0; o < numOutput; ++o)
                hoWeights[j][o] = weights[k++];
        for (int o = 0; o < numOutput; ++o)
            oBiases[o] = weights[k++];
    }


    public double[] ComputeOutputs(double[] xValues)
    {
        Array.Copy(xValues, inputs, numInput);

        for (int j = 0; j < numHidden; ++j)
        {
            double sum = hBiases[j];
            for (int i = 0; i < numInput; ++i)
                sum += inputs[i] * ihWeights[i][j];
            hOutputs[j] = Math.Tanh(sum);
        }

        for (int k = 0; k < numOutput; ++k)
        {
            double sum = oBiases[k];
            for (int j = 0; j < numHidden; ++j)
                sum += hOutputs[j] * hoWeights[j][k];
            outputs[k] = sum;
        }

        var result = new double[numOutput];
        Array.Copy(outputs, result, numOutput);
        return result;
    }

    public double[] PredictSingleOutput(double[][] data)
    {
        if (numOutput != 1)
            throw new InvalidOperationException("PredictSingleOutput requires numOutput == 1.");

        var xValues = new double[numInput];
        var predictions = new double[data.Length];
        for (int i = 0; i < data.Length; ++i)
        {
            Array.Copy(data[i], xValues, numInput);
            predictions[i] = ComputeOutputs(xValues)[0];
        }
        return predictions;
    }


    public List<EpochRecord> Train(double[][] trainData, double[][]? validationData,
        int maxEpochs, double learnRate, double momentum, double weightDecay,
        int logInterval = 0, TextWriter? log = null)
    {
        var oSignals = new double[numOutput];
        var hSignals = new double[numHidden];

        double[][] ihPrevDelta = MakeMatrix(numInput, numHidden);
        double[] hbPrevDelta = new double[numHidden];
        double[][] hoPrevDelta = MakeMatrix(numHidden, numOutput);
        double[] obPrevDelta = new double[numOutput];

        var xValues = new double[numInput];
        var tValues = new double[numOutput];

        var order = new int[trainData.Length];
        for (int i = 0; i < order.Length; ++i) order[i] = i;

        var history = new List<EpochRecord>(maxEpochs);

        for (int epoch = 1; epoch <= maxEpochs; ++epoch)
        {
            Shuffle(order);

            foreach (int idx in order)
            {
                Array.Copy(trainData[idx], xValues, numInput);
                Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
                ComputeOutputs(xValues);

                for (int k = 0; k < numOutput; ++k)
                    oSignals[k] = tValues[k] - outputs[k];

                for (int j = 0; j < numHidden; ++j)
                {
                    double sum = 0.0;
                    for (int k = 0; k < numOutput; ++k)
                        sum += oSignals[k] * hoWeights[j][k];
                    hSignals[j] = (1.0 - hOutputs[j] * hOutputs[j]) * sum;
                }

                for (int j = 0; j < numHidden; ++j)
                {
                    for (int k = 0; k < numOutput; ++k)
                    {
                        double gradient = oSignals[k] * hOutputs[j];
                        double delta = learnRate * (gradient - weightDecay * hoWeights[j][k]);
                        hoWeights[j][k] += delta + momentum * hoPrevDelta[j][k];
                        hoPrevDelta[j][k] = delta;
                    }
                }
                for (int k = 0; k < numOutput; ++k)
                {
                    double delta = learnRate * oSignals[k];
                    oBiases[k] += delta + momentum * obPrevDelta[k];
                    obPrevDelta[k] = delta;
                }

                for (int i = 0; i < numInput; ++i)
                {
                    for (int j = 0; j < numHidden; ++j)
                    {
                        double gradient = hSignals[j] * inputs[i];
                        double delta = learnRate * (gradient - weightDecay * ihWeights[i][j]);
                        ihWeights[i][j] += delta + momentum * ihPrevDelta[i][j];
                        ihPrevDelta[i][j] = delta;
                    }
                }
                for (int j = 0; j < numHidden; ++j)
                {
                    double delta = learnRate * hSignals[j];
                    hBiases[j] += delta + momentum * hbPrevDelta[j];
                    hbPrevDelta[j] = delta;
                }
            }

            double trainMse = MeanSquaredError(trainData);
            double validationMse = validationData is null
                ? double.NaN
                : MeanSquaredError(validationData);
            history.Add(new EpochRecord(epoch, trainMse, validationMse));

            if (log is not null && logInterval > 0 && (epoch == 1 || epoch % logInterval == 0))
                log.WriteLine($"epoch = {epoch,5}  train MSE = {trainMse:F5}  validation MSE = {validationMse:F5}");
        }

        return history;
    }


    public double MeanSquaredError(double[][] data)
    {
        var xValues = new double[numInput];
        var tValues = new double[numOutput];
        double sumSquaredError = 0.0;

        foreach (double[] row in data)
        {
            Array.Copy(row, xValues, numInput);
            Array.Copy(row, numInput, tValues, 0, numOutput);
            double[] y = ComputeOutputs(xValues);
            for (int k = 0; k < numOutput; ++k)
            {
                double error = tValues[k] - y[k];
                sumSquaredError += error * error;
            }
        }
        return sumSquaredError / (data.Length * numOutput);
    }

    private void Shuffle(int[] sequence)
    {
        for (int i = sequence.Length - 1; i > 0; --i)
        {
            int j = rnd.Next(i + 1);
            (sequence[i], sequence[j]) = (sequence[j], sequence[i]);
        }
    }
}