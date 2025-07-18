# ALUNO VICTOR AUGUSTO COSTA MONTEIRO
# NUSP 8942937
import numpy as np  # matrix calculations


def feed_forward(inputs, targets,
                 weights_hidden, weights_output,
                 perceptrons_hidden, perceptrons_output,
                 function):
    # some auxiliary structures
    # matrix filled with ones, results of hidden layer processing
    outputs_hidden = np.ones((len(inputs), perceptrons_hidden + 1))
    # results of final layer processing
    outputs_final = np.ones((len(targets), len(targets[0])))

    # for each perceptron in hidden layer
    for perceptron in range(perceptrons_hidden):
        # i = perceptron + 1 # skip bias input for hidden outputs
        # for each training pair
        for row in range(len(inputs)):
            # dot product input and weights
            z_in = np.dot(inputs[row], weights_hidden[:, perceptron])
            # ACTIVATION FUNCTION
            z = activation_function(z_in, function)
            # update hidden outputs
            outputs_hidden[row][perceptron] = z
    # for each perceptron in output layer
    for perceptron in range(perceptrons_output):
        # for each hidden output pair
        for row in range(len(outputs_hidden)):
            # dot product input and weights
            y_in = np.dot(outputs_hidden[row], weights_output[:, perceptron])
            # ACTIVATION FUNCTION
            y = activation_function(y_in, function)
            # update final outputs
            outputs_final[row][perceptron] = y
    return outputs_hidden, outputs_final


def training(inp, tar, rate,
             perceptrons_hidden, max_epochs,
             early_stop, function, patience):
    inputs = inp
    targets = tar
    inputs_early_stop = []
    targets_early_stop = []
    last_validation_loss = float('inf')
    best_validation_loss = float('inf')
    patience_counter = 0
    perceptrons_output = len(targets[0])
    # randomized weights to start, [-0.5,0.5]
    weights_hidden = np.random.uniform(-0.5, 0.5, (len(inputs[0]), perceptrons_hidden))
    weights_output = np.random.uniform(-0.5, 0.5, (perceptrons_hidden + 1, perceptrons_output))
    start_weights = [weights_hidden, weights_output]

    stopped_early = False
    if early_stop:
        # Training data and testing data, +- 80/20
        # evening classes between training and testing
        alphabet_repetition = len(inputs) / perceptrons_output
        split_index = int(alphabet_repetition * 0.8) * perceptrons_output
        # train data
        inputs = inp[:split_index]
        targets = tar[:split_index]
        # test data
        inputs_early_stop = inp[split_index:]
        targets_early_stop = tar[split_index:]

    # start of epochs training
    epoch_counter = 0
    while epoch_counter < max_epochs:
        # if epoch_counter == 1: break # to compare with handmade first.md epoch
        epoch_counter += 1
        print(f"Running epoch {epoch_counter} of {max_epochs}")
        outputs_hidden, outputs_final = feed_forward(inputs, targets,
                                                     weights_hidden, weights_output,
                                                     perceptrons_hidden, perceptrons_output, function)
        # backpropagation
        for t in range(len(targets)):
            # output layer
            error_info_output = np.zeros((1, perceptrons_output))
            weights_output_delta = np.zeros_like(weights_output)
            for m in range(perceptrons_output):
                y_in = np.dot(outputs_hidden[t], weights_output[:, m])
                error_info_output[0][m] = ((targets[t][m] - outputs_final[t][m])
                                           * activation_function_derivative(y_in, function))
                for z in range(len(outputs_hidden[t])):
                    weights_output_delta[z][m] = rate * error_info_output[0][m] * outputs_hidden[t][z]
            # hidden layer
            error_info_hidden = np.zeros((1, perceptrons_hidden))
            weights_hidden_delta = np.zeros_like(weights_hidden)
            for p in range(perceptrons_hidden):
                outer_info = 0
                for m in range(perceptrons_output):
                    outer_info += weights_output[p][m] * error_info_output[0][m]
                z_in = np.dot(inputs[t], weights_hidden[:, p])
                error_info_hidden[0][p] = outer_info * activation_function_derivative(z_in, function)
                for x in range(len(inputs[0])):
                    weights_hidden_delta[x][p] = rate * error_info_hidden[0][p] * inputs[t][x]
            # weight change
            weights_hidden += weights_hidden_delta
            weights_output += weights_output_delta
        # early stop
        if early_stop:
            # feed forward
            val_outputs_hidden, val_outputs_final = feed_forward(inputs_early_stop, targets_early_stop,
                                                                 weights_hidden, weights_output,
                                                                 perceptrons_hidden, perceptrons_output, function)
            # mean squared error
            validation_loss = np.mean((targets_early_stop - val_outputs_final) ** 2)
            print(f"    Validation loss: {validation_loss}")

            if validation_loss < last_validation_loss:
                patience_counter = 0
            else:
                patience_counter += 1
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
            last_validation_loss = validation_loss
            print(f"    Patience counter: {patience_counter}")
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch_counter}")
                stopped_early = True
                break
    final_weights = [weights_hidden, weights_output]

    return [start_weights, final_weights, epoch_counter, best_validation_loss, stopped_early]


def activation_function(x, func):
    # Sigmoid function BINARY
    if func == 'sigmoid-binary':
        return 1 / (1 + np.exp(-x))


def activation_function_derivative(x, func):
    if func == 'sigmoid-binary':
        return activation_function(x, func) * (1 - activation_function(x, func))


def confusion_matrix(y_true, y_pred, classes, threshold):
    y_pred = apply_threshold(y_pred, threshold)
    conf_matrix = np.zeros((classes + 1, classes + 1), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        true_index = one_hot_index(true_label)
        pred_index = one_hot_index(pred_label)

        if true_index == -1 or pred_index == -1:
            if true_index == -1:
                print("True index com erro(?)")
                continue
            elif pred_index == -1:
                conf_matrix[-1, true_index] += 1
                continue
        else:
            conf_matrix[pred_index, true_index] += 1
    return conf_matrix


def apply_threshold(y_pred_prob, threshold):

    y_pred = (y_pred_prob >= threshold).astype(int)
    return y_pred


def one_hot_index(one_hot):
    # one-hot vectors, one element equals 1 and the rest equals 0
    if np.sum(one_hot) == 1:
        return np.argmax(one_hot)  # max value index in array
    return -1
