import pandas as pd # biblioteca para ler e escrever arquivos csv
import numpy as np # matrix calculations

def training(inputs, targets, rate, hiddenPerceptrons, outputPerceptrons):
    weightsHidden = np.array([[-0.1,-0.1,0.1], [0.1,0.1,-0.1], [-0.1,0.1,-0.1]], dtype=np.float64)
    weightsOutput = np.array([[-0.1,0.1], [0.1,-0.1], [0,0.1], [0.1,-0.1]], dtype=np.float64)
    # print("Pre training")
    # print("Inputs matrix"), print(inputs)
    # print("Targets matrix"), print(targets)
    # print("Inputs hidden matrix"), print(outputsHid)
    # print("Output weights matrix"), print(weightsOutput)
    # print("Hidden weights matrix"), print(weightsHidden)
    # print("Delta output weights matrix"), print(weightsDeltaOutput)
    # print("Delta hidden weights matrix"), print(weightsDeltaHidden)
    # start of epochs training
    epochCount = 0
    while True: 
        # epoch counter
        epochCount = epochCount + 1
        # some auxialiary structures defined for every epoch
        outputsHid = inputs.copy() # results of hidden layer processing
        outputsOut = targets.copy() # results of hidden layer processing
        weightsDeltaOutput = weightsOutput.copy() # delta error for backprogation error on output layer
        weightsDeltaHidden = weightsHidden.copy() # delta error for backprogation error on hidden layer
        # for each perceptron in hidden layer
        for perceptron in range(len(weightsHidden[0])):
            # for each training pair
            for row in range(len(inputs)):
                # dot product input and weights
                y_in = np.dot(inputs[row],weightsHidden[:,perceptron])
                # ACTIVATION FUNCTION
                y = activationFunction(y_in)
                # update hidden outputs
                outputsHid[row][perceptron] = y
        # add bias to hidden layer output
        outputsHid = np.insert(outputsHid, 0, [1], axis=1)

        # for each perceptron in output layer
        for perceptron in range(len(weightsOutput[0])):
            # for each hidden output pair
            for row in range(len(outputsHid)):
                # dot product input and weights
                y_in = np.dot(outputsHid[row],weightsOutput[:,perceptron])
                # ACTIVATION FUNCTION
                y = activationFunction(y_in)
                # update final outputs
                outputsOut[perceptron] = y
                
        print(f"Epoch Count: {epochCount}")
        epochLimit = 1000 * 10
        # check exit conditions, either correct answer or epoch limit
        if np.array_equal(outputsOut, targets) or epochCount == epochLimit: 
            if epochCount == epochLimit: print("Did not converge!")
            break
        else:
            # backpropagation algorithm for output layer
            infos_error_output = targets.copy()
            for perceptron in range(len(outputsOut)):
                # info error for output layer
                error_info = (targets[perceptron] - outputsOut[perceptron])*activationFunctionDerivative(y_in)
                infos_error_output[perceptron] = error_info
                for w in range(len(weightsOutput)):
                    delta_error_output = rate * error_info * outputsHid[row][w]
                    weightsDeltaOutput[w][perceptron] = delta_error_output
            # backpropagation algorithm for hidden layer
            infos_error_hidden = np.eye(3,1)
            # first.md the info related to the outter layer
            for w in range(len(weightsOutput) - 1):
                a = w + 1 # need a workaround cause bias is the first.md in the matrix
                outer_info = np.dot(weightsOutput[a], infos_error_output)
                infos_error_hidden[w] = outer_info
            # now the info related to the layer itself
            for perceptron in range(len(weightsHidden[0])):
                z_in = np.dot(inputs[0], weightsHidden[:,perceptron])
                derivative = activationFunctionDerivative(z_in)
                infos_error_hidden[perceptron] = infos_error_hidden[perceptron] * derivative
            # get deltas for hidden weights
            for info in range(len(infos_error_hidden)):
                for delta in range(len(weightsDeltaHidden)):
                    error_delta_hidden = rate * infos_error_hidden[info] * inputs[0][perceptron]
                    weightsDeltaHidden[delta][info] = error_delta_hidden
            # update weights
            weightsHidden = weightsHidden + weightsDeltaHidden
            # print("New Weights hidden"), print(weightsHidden)
            weightsOutput = weightsOutput + weightsDeltaOutput
            # print("New Weights output"), print(weightsOutput)         
        
    # print("After training")
    # print("Inputs matrix"), print(inputs)
    # print("Targets matrix"), print(targets)
    # print("Inputs hidden matrix"), print(outputsHid)
    # print("Output weights matrix"), print(weightsOutput)
    # print("Hidden weights matrix"), print(weightsHidden)
    # print("Delta output weights matrix"), print(weightsDeltaOutput)
    # print("Delta hidden weights matrix"), print(weightsDeltaHidden)

def activationFunction(input):
    # Sigmoid function
    return 1/(1 + np.exp(-input))

def activationFunctionDerivative(input):
    return activationFunction(input)*(1 - activationFunction(input))

if __name__ == "__main__":
    filePath = "/ep1-mlp/portas-logicas/problemXOR.csv"
    inputsMatrix = np.array([[1,1,1]], dtype=np.float64)
    targetsMatrix = np.array([[1],[0]], dtype=np.float64)
    rate = 0.5
    hiddenPerceptrons = 3
    outputPerceptrons = 2
    training(inputsMatrix, targetsMatrix, rate, hiddenPerceptrons, outputPerceptrons)
