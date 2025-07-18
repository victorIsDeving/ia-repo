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

def training(inputs, targets, rate, hiddenPerceptrons, outputPerceptrons, paradaAntecipada):
    # randomized weights to start, [-0.5,0,5]
    weightsHidden = np.random.uniform(-0.5, 0.5, (len(inputs[0]), hiddenPerceptrons)) # bias considered inside input
    weightsOutput = np.random.uniform(-0.5, 0.5, (hiddenPerceptrons + 1, outputPerceptrons)) # added bian for output layer
    # some auxialiary structures 
    outputsHid = np.ones((len(inputs), hiddenPerceptrons)) # matrix filled with ones, results of hidden layer processing
    outputsHid = np.insert(outputsHid, 0, 1, axis=1) # add bias to hidden layer output
    outputsOut = np.ones((len(targets), len(targets[0]))) # results of final layer processing
    
    # start of epochs training
    epochCounter = 0
    epochLimit = 1000 * 15
    while epochCounter < epochLimit: 
        # if epochCounter == 1: break # to compare with handmade first.md epoch
       
        # epoch counter
        epochCounter += 1

        # for each perceptron in hidden layer
        for perceptron in range(hiddenPerceptrons):
            i = perceptron + 1 # skip bias input for hidden outputs 
            # for each training pair
            for row in range(len(inputs)):
                # dot product input and weights
                z_in = np.dot(inputs[row],weightsHidden[:,perceptron])
                # ACTIVATION FUNCTION
                z = activationFunction(z_in)
                # update hidden outputs
                outputsHid[row][i] = z   

        # for each perceptron in output layer
        for perceptron in range(outputPerceptrons):
            # for each hidden output pair
            for row in range(len(outputsHid)):
                # dot product input and weights
                y_in = np.dot(outputsHid[row],weightsOutput[:,perceptron])
                # ACTIVATION FUNCTION
                y = activationFunction(y_in)
                # update final outputs
                outputsOut[row][perceptron] = y
                
        # change only for wrong answers
        for t in range(len(targets)):
            # Output layer
            info_error_output = np.zeros((1, outputPerceptrons))
            weightsDeltaOutput = np.zeros_like(weightsOutput)
            for m in range(outputPerceptrons):
                y_in = np.dot(outputsHid[t], weightsOutput[:, m])
                error_info_output = (targets[t][m] - outputsOut[t][m]) * activationFunctionDerivative(y_in)
                info_error_output[0][m] = error_info_output
                for z in range(len(outputsHid[t])):
                    weightsDeltaOutput[z][m] = rate * info_error_output[0][m] * outputsHid[t][z]

            # Hidden layer
            infos_error_hidden = np.zeros((1, hiddenPerceptrons))
            weightsDeltaHidden = np.zeros_like(weightsHidden)
            for p in range(hiddenPerceptrons):
                outer_info = 0
                for m in range(outputPerceptrons):
                    outer_info += weightsOutput[p + 1][m] * info_error_output[0][m]
                z_in = np.dot(inputs[t], weightsHidden[:, p])
                error_info_hidden = outer_info * activationFunctionDerivative(z_in)
                infos_error_hidden[0][p] = error_info_hidden
                for x in range(len(inputs[0])):
                    weightsDeltaHidden[x][p] = rate * infos_error_hidden[0][p] * inputs[t][x]

            weightsHidden += weightsDeltaHidden
            weightsOutput += weightsDeltaOutput
        
    print(f"After training, epochs = {epochCounter}")
    print("outputsHid"), print(outputsHid)
    print("outputsOut"), print(outputsOut)
    print("weightsHidden"), print(weightsHidden)
    print("weightsOutput"), print(weightsOutput)

def activationFunction(input):
    # Sigmoid function BINARY
    return 1/(1 + np.exp(-input))

def activationFunctionDerivative(input):
    return activationFunction(input)*(1 - activationFunction(input))

def erroQuadraticoMedio(targets, outputs):
    return np.mean((targets - outputs) ** 2)

if __name__ == "__main__":
    filePath = "/ep1-mlp/portas-logicas/problemXOR.csv"
    inputsMatrix = readInput(filePath)
    targetsMatrix = createTargetsMatrix(inputsMatrix)
    inputsMatrix = biasToInputs(inputsMatrix)
    rate = 0.5
    hiddenPerceptrons = 2
    outputPerceptrons = 1
    earlyStop = False
    crossValidation = False

    if crossValidation:
        # split data
        training(inputsMatrix, targetsMatrix, rate, hiddenPerceptrons, outputPerceptrons, earlyStop)
    else:
        # regular data
        training(inputsMatrix, targetsMatrix, rate, hiddenPerceptrons, outputPerceptrons, earlyStop)
