import pandas as pd # read and wirte .csv files
import numpy as np # matrix calculations
import random # randomize some choices

# Reads input csv file and returns a matrix of inputs + target
def readInput(file) :
    matrix = []
    with open(file, newline='') as csvFile:
        df = pd.read_csv(csvFile, header=None)
        for row in range(df.shape[0]):
            array = df.values[row].tolist()
            matrix.append(array)
    return np.array(matrix)

# Divides the inputs matrix into inputs and targets matrixes
def createTargetsMatrix(inputMatrix):
    matrix = []
    for row in range(len(inputMatrix)):
        array = []
        target = inputMatrix[row][-1]
        array.append(target)
        matrix.append(array)
    return np.array(matrix)

# Adds bias input to input matrix
def biasToInputs(inputMatrix):
    # Remove last column
    inputMatrix = inputMatrix[:,:-1]
    # Add a column with all 1s for bias
    new_column = np.full((inputMatrix.shape[0], 1), 1)
    # Concat matrixes
    inputMatrix = np.hstack((new_column, inputMatrix))
    return inputMatrix

def training(inputs, targets, rate, hiddenPerceptrons, outputPerceptrons):
    weightsHidden = np.random.uniform(-0.5, 0.5, (len(inputs[0]), hiddenPerceptrons))
    weightsOutput = np.random.uniform(-0.5, 0.5, (hiddenPerceptrons + 1, outputPerceptrons))
    
    outputsHid = np.ones((len(inputs), hiddenPerceptrons + 1)) # +1 for bias
    outputsOut = np.ones((len(targets), len(targets[0])))

    epochCounter = 0
    epochLimit = 15000
    
    while epochCounter < epochLimit:
        epochCounter += 1

        # Forward pass
        for t in range(len(inputs)):
            # Hidden layer
            for perceptron in range(hiddenPerceptrons):
                z_in = np.dot(inputs[t], weightsHidden[:, perceptron])
                outputsHid[t][perceptron + 1] = activationFunction(z_in)  # +1 to skip bias

            # Output layer
            for perceptron in range(outputPerceptrons):
                y_in = np.dot(outputsHid[t], weightsOutput[:, perceptron])
                outputsOut[t][perceptron] = activationFunction(y_in)

        # Backpropagation
        for t in range(len(targets)):
            if not np.array_equal(targets[t],outputsOut[t]): # compare all elements for given result
                # backpropagation algorithm for output layer
                # auxiliary structures
                info_error_output = np.zeros((1, outputPerceptrons)) 
                weightsDeltaOutput = weightsOutput.copy() # delta error for backprogation error on output layer
                for m in range(outputPerceptrons): # for each perceptron output
                    # info error for output layer
                    y_in = y_in_matrix[0][m]
                    error_info_output = (targets[t][m] - outputsOut[t][m])*activationFunctionDerivative(y_in)
                    info_error_output[0][m] = error_info_output
                    for z in range(len(outputsHid[0])):
                        delta_error_output = rate * info_error_output[0][m] * outputsHid[t][z]
                        weightsDeltaOutput[z][m] = delta_error_output

                # backpropagation algorithm for hidden layer
                # auxiliary structures
                infos_error_hidden = np.zeros((1, hiddenPerceptrons))
                weightsDeltaHidden = weightsHidden.copy() # delta error for backprogation error on hidden layer
                for p in range(hiddenPerceptrons):
                    # first.md the info related to the outter layer, the sigma sum in the information error
                    outer_info = 0
                    h = p + 1 # ignore first.md line in weightsOutput as it is for bias
                    for m in range(outputPerceptrons): # for each final perceptron
                        # for each hidden perceptron you take account every weight he carried to the next layer
                        outer_info += weightsOutput[h][m] * info_error_output[0][m] 
                    z_in = z_in_matrix[0][p]
                    error_info_hidden = outer_info * activationFunctionDerivative(z_in)
                    infos_error_hidden[0][p] = error_info_hidden
                    for x in range(len(inputs[0])):
                        error_delta_hidden = rate * infos_error_hidden[0][p] * inputs[t][x]
                        weightsDeltaHidden[x][p] = error_delta_hidden
                # update weights
                weightsHidden += weightsDeltaHidden
                weightsOutput += weightsDeltaOutput

    print(f"After training, epochs = {epochCounter}")
    print("weightsHidden"), print(weightsHidden)
    print("weightsOutput"), print(weightsOutput)

    # Test the final outputs
    for t in range(len(inputs)):
        for perceptron in range(hiddenPerceptrons):
            z_in = np.dot(inputs[t], weightsHidden[:, perceptron])
            outputsHid[t][perceptron + 1] = activationFunction(z_in)  # +1 to skip bias
        for perceptron in range(outputPerceptrons):
            y_in = np.dot(outputsHid[t], weightsOutput[:, perceptron])
            outputsOut[t][perceptron] = activationFunction(y_in)
    print("Final outputs"), print(outputsOut)

def activationFunction(input):
    # Sigmoid function BINARY
    return 1/(1 + np.exp(-input))

def activationFunctionDerivative(input):
    sig = activationFunction(input)
    return sig*(1 - sig)

if __name__ == "__main__":
    filePath = "/ep1-mlp/portas-logicas/problemXOR.csv"
    inputsMatrix = readInput(filePath)
    targetsMatrix = createTargetsMatrix(inputsMatrix)
    inputsMatrix = biasToInputs(inputsMatrix)
    rate = 0.5
    hiddenPerceptrons = 2
    outputPerceptrons = 1
    # inputsMatrix = np.array([[1.0, 1.0, 1.0]])
    # targetsMatrix = np.array([[1.0, 0.0]])
    
    training(inputsMatrix, targetsMatrix, rate, hiddenPerceptrons, outputPerceptrons)
