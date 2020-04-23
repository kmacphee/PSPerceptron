class Perceptron {
    
    # An array of ints tracking the number of misclassifications per epoch.
    [array]         $ErrorsPerEpoch = @()
    # An array of doubles that hold the weight coefficients for each feature in a sample.
    [array]         $Weights        = @()
    # A constant used for calculating the net input of an individual sample.
    hidden [double] $BiasValue      = 0.0
    # The number of passes over the dataset during training.
    hidden [int]    $Epochs         = 50
    # Controls the size of weight updates during training.
    hidden [double] $LearningRate   = 0.01
    # The seed for the random number generator used to initialize the weights vector.
    hidden [int]    $RandomSeed     = 1
    
    # The default constructor just uses the default values of properties.
    Perceptron() {}
    
    Perceptron([double] $LearningRate, [int] $Epochs, [int] $RandomSeed) {
        $this.LearningRate = $LearningRate
        $this.Epochs = $Epochs
        $this.RandomSeed = $RandomSeed
    }
    
    # Trains the Perceptron with the given sample matrix and vector (array) of target
    # class labels. The target class labels are the labels that the Perceptron needs
    # to learn to classify when given unlabelled data after training. Training involves
    # building a vector (array) of weight coefficients that can be multipled with the
    # feature values of the sample and passed to a decision making function (see: Classify)
    # for classification.
    [Void] Train([double[,]] $Samples, [int[]] $Targets) {
        if ($Samples.GetLength(0) -le 1) {
            throw "No training samples provided."
        }
        if ($Targets.length -ne $Samples.GetLength(0)) {
            throw "Number of target class labels does not equal number of samples."
        }
        $numFeatures = $Samples.GetLength(1)
        if ($numFeatures -le 1) {
            throw "No feature values provided in training samples."
        }
        
        # Create the vector (array) of weight coefficients for the number of features
        # in the sample. Initialize the weight values with random numbers that are
        # close to the decision boundary (0.0).
        $this.Weights = New-Object -TypeName "double[]" -ArgumentList $numFeatures
        $null = Get-Random -SetSeed $this.RandomSeed
        for ($i = 0; $i -lt $this.Weights.length; $i++) {
            $this.Weights[$i] = Get-Random -Minimum -0.01 -Maximum 0.01  
        }
        
        $this.ErrorsPerEpoch = [System.Collections.Generic.List[int]]::new()
        # For each epoch (pass over the dataset).
        for ($i = 0; $i -lt $this.Epochs; $i++) {
            $errors = 0
            # For each sample in the dataset.
            for ($j = 0; $j -lt $Samples.GetLength(0); $j++) {
                $sample = New-Object -TypeName "double[]" -ArgumentList $numFeatures
                # Copy sample features into a one-dimensional array to represent a
                # single sample.
                for ($k = 0; $k -lt $numFeatures; $k++) {
                    $sample[$k] = $Samples[$j, $k]
                }
                
                $target = $Targets[$j]
                
                # Attempt to classify the sample and compare the result to the target
                # value to produce an update value. If 'result' equals 'target' (i.e.
                # we classified the sample correctly), 'target' minus 'result' will
                # equal zero, thus the update value will be zero and the weights will
                # remain unchanged. If 'result' does not equal 'target', 'update' will
                # be a number that will push the weight coefficients towards 0 or 1,
                # depending on what the result should have been.
                $result = $this.Classify($sample)
                $update = $this.LearningRate * ($target - $result)
                
                # Update all weight coefficients and the bias value.
                for ($k = 0; $k -lt $this.Weights.length; $k++) {
                    $this.Weights[$k] += ($update * $sample[$k])
                }
                $this.BiasValue += $update
                
                if ($update -ne 0) {
                    $errors++
                }
            }
            # Keep track of the number of misclassifications in each epoch. If the
            # perceptron is learning correctly this number should reduce over the
            # total number of epochs.
            $this.ErrorsPerEpoch.Add($errors)
        }
    }
    
    # The decision function of the Perceptron. If the net input of the sample is
    # greater than 0.0 it will return one class label, otherwise it will return
    # the other.
    [int] Classify([array] $Sample) {
        if ($this.getNetInput($Sample) -gt 0.0) {
            return 1
        }
        
        return 0
    }

    # Get the net input of the given sample. The net input converts the entire
    # feature vector (array) into a single number. This single number is the
    # dot product of the feature vector and the weights vector, plus the bias
    # value.
    hidden [double] getNetInput([array] $Sample) {
        if ($Sample.length -ne $this.Weights.length) {
            throw "Invalid number of features in the sample."
        }
        
        $dotProduct = 0.0
        for ($i = 0; $i -lt $Sample.length; $i++) {
            $dotProduct += ($Sample[$i] * $this.Weights[$i])
        }
        
        return $dotProduct + $this.BiasValue
    }
}
