import pandas as pd #biblioteca para ler e escrever arquivos csv
import numpy as np #matrix calculations
import copy

#Reads input csv file and returns a matrix of inputs + target
def readInput(file) :
    matrix = []
    with open(file, newline='') as csvFile:
        df = pd.read_csv(csvFile, header=None)
        for row in range(df.shape[0]):
            array = df.values[row].tolist()
            matrix.append(array)
    return matrix

#Divides the inputs matrix into inputs and targets matrixes
def createTargetsMatrix(inputMatrix):
    matrix = []
    for row in range(len(inputMatrix)):
        array = []
        target = inputMatrix[row].pop()
        inputMatrix[row].append(1) #add the bias input
        array.append(target)
        matrix.append(array)  
    return matrix

#Creates weights for each variable
def createWeightsMatrix(inputMatrix):
    matrix = []
    for column in range(len(inputMatrix[0])):
        array = [1]
        matrix.append(array)        
    return matrix


def training(inputs, targets, weights, rate, threshold):
    #initialize weights and bias, all equals zero for simplicity
    for row in range(len(weights)):
        for column in range(len(weights[0])):
            weights[row][column] = weights[row][column] * 0
    #initialize learning rate
    rate = rate * 1
    #start of epochs training
    epochCount = 0
    while True:
        #flag to check for weights change in epoch
        weightsChange = False
        #for each training pair
        for row in range(len(inputs)):
            #dot product input and weights
            y_in = np.dot(inputs[row],weights)
            #ACTIVATION FUNCTION
            y = activationFunction(y_in[0], threshold)
            #update weights and bias
            if y != targets[row][0]:
                weightsChange = True
                for w in range(len(weights)):
                    weights[w][0] = weights[w][0] + rate * targets[row][0] * inputs[row][w]
        epochCount = epochCount + 1
        #conditions to exit processing, either by converging either by epoch reaching limit
        if weightsChange == False or epochCount == 100: 
            if epochCount == 100: print("Did not converge!")
            break
    print(f"Epoch Count: {epochCount}")

def activationFunction(input, threshold):
    #step function
    output = 1
    if input < threshold: output = -1
    return output

if __name__ == "__main__":
    filePath = "/ep1-mlp/portas-logicas/problemAND.csv"
    # filePath = "C:\sistemas-informacao\ia-ACH2016\ep1-mlp\portas-logicas\problemOR.csv"
    #shouldnt work for XOR problem
    # filePath = "C:\sistemas-informacao\ia-ACH2016\ep1-mlp\portas-logicas\problemXOR.csv"
    inputsMatrix = readInput(filePath)
    targetsMatrix = createTargetsMatrix(inputsMatrix)
    weightsMatrix = createWeightsMatrix(inputsMatrix)
    rate = 1
    threshold = 0.2
    print("Pre training")
    print(inputsMatrix)
    print(weightsMatrix)
    print(targetsMatrix)
    training(inputsMatrix, targetsMatrix, weightsMatrix, rate, threshold)
    print("After training")
    print(inputsMatrix)
    print(weightsMatrix)
    print(targetsMatrix)