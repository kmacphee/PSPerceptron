using module ./Perceptron.psm1

# Acquire Iris dataset and import from CSV file.
Invoke-WebRequest -Uri 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' -OutFile './iris.data'
$samples = Import-Csv -Path './iris.data' -Header 'SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'ClassName'

# Select only setosa and versicolor samples (first 100). Our Perceptron is only a binary classifier.
$samples = $samples | Select-Object -First 100

# Turn sample data into a two-dimensional array of feature values and an array of target class 
# labels (integers) that we will use in training.
$samples2dArray = New-Object "double[,]" -ArgumentList $samples.length, 4
$targetClassLabels = New-Object "int[]" -ArgumentList $samples.length
for ($i = 0; $i -lt $samples.length; $i++) {
    $samples2dArray[$i, 0] = $samples[$i].SepalLength
    $samples2dArray[$i, 1] = $samples[$i].SepalWidth
    $samples2dArray[$i, 2] = $samples[$i].PetalLength
    $samples2dArray[$i, 3] = $samples[$i].PetalWidth

    if ($samples[$i].ClassName -eq 'Iris-setosa') {
        $targetClassLabels[$i] = 0 # setosa = 0
    }
    else {
        $targetClassLabels[$i] = 1 # versicolor = 1
    }
}

# Create our Perceptron object. Experimenting with the arguments will give it different
# performance characteristics.
$perceptronArgs = @{
    "LearningRate" = 0.0001;
    "Epochs" = 10;
    "RandomSeed" = 1
}
$perceptron = New-Object -TypeName Perceptron -Property $perceptronArgs

# Train the Perceptron with the sample data. Output the number of errors encountered
# per pass over the sample data.
$perceptron.Train($samples2dArray, $targetClassLabels)
Write-Host "Errors per epoch: $($perceptron.ErrorsPerEpoch)"

# If we feed the Perceptron new sample data, it should correctly identify the type of Iris.
$newSetosaSample = @(5.1, 3.5, 1.4, 0.3)
$result = $perceptron.Classify($newSetosaSample)
if ($result -eq 0) {
    Write-Host "Successfully classified a Setosa Iris."
}
$newVersicolorSample = @(6.7, 3.0, 5.0, 1.7)
$result = $perceptron.Classify($newVersicolorSample)
if ($result -eq 1) {
    Write-Host "Successfully classified a Versicolor Iris."
}
