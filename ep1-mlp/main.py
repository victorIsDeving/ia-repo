# ALUNO VICTOR AUGUSTO COSTA MONTEIRO
# NUSP 8942937

# This Python file uses the following encoding: utf-8
import numpy as np
import os
import time
from training import training, feed_forward, confusion_matrix
from report import markdown_report, dict_resume


start_time = time.time()

data_path = "C:/sistemas-informacao/ia-ACH2016/ep1-mlp/COMPLETO/X.npy"
target_path = "C:/sistemas-informacao/ia-ACH2016/ep1-mlp/COMPLETO/Y_classe.npy"
# configs do resultado
# output folder
# - 00: testes com muitas epocas e feitos com várias configs diferentes
# - 01: testes so com parada antecipada
# - 02: testes so com validação cruzada
# - 03: testes com parada antecipada e validação cruzada
# - testes-apresentação: testes feitos durante a gravação do vídeo
output_folder = "testes-apresentacao"
# Dentro de cada output folder tem uma pasta para cada teste feito com uma config diferente
test = "Com"
# criacao dos diretorios para cada teste
output_file = f"report{test}.md"
output_dir = f"test{test}"
output_loc = f"outputs/{output_folder}/{output_dir}/"
if not os.path.exists(f"C:/sistemas-informacao/ia-ACH2016/ep1-mlp/outputs/{output_folder}"):
    os.makedirs(f"C:/sistemas-informacao/ia-ACH2016/ep1-mlp/outputs/{output_folder}")
if not os.path.exists(f"C:/sistemas-informacao/ia-ACH2016/ep1-mlp/{output_loc}"):
    os.makedirs(f"C:/sistemas-informacao/ia-ACH2016/ep1-mlp/{output_loc}")

# configs do teste
max_epochs = 200
hidden_perceptrons = 90
output_perceptrons = 26  # len(targets[0]), total classes
learn_rate = 0.8
threshold = 0.55
act_func = 'sigmoid-binary'
early_stop_bool = True
early_stop_patience = 5
cross_validation_bool = True
cross_validation_folds = 4

# inputs
data = np.load(data_path)
input_data = np.random.rand(len(data), 120)
for i, row in enumerate(data):
    input_data[i] = row.flatten()
    np.append(input_data[i], 1)  # adding bias

# targets
target = np.load(target_path)
input_target = np.random.rand(len(target), 26)
for i, row in enumerate(target):
    input_target[i] = row.flatten()

# evening classes between training and testing
alphabet_repetition = len(input_data) / output_perceptrons
if cross_validation_bool:
    # training data and testing data, +- 80/20
    split_index = int(alphabet_repetition * 1 / cross_validation_folds) * output_perceptrons
    # save every fold results
    start_weights_hidden_list = []
    start_weights_out_list = []
    final_weights_hidden_list = []
    final_weights_out_list = []
    epoch_counters = []
    validation_losses = []
    output_final = []
    output_hidden = []
    all_test_targets = []
    all_test_predictions = []

    for i in range(cross_validation_folds):
        print(f"Fold {i + 1} de {cross_validation_folds}")
        # test data
        test_data = input_data[split_index * i:split_index * (i + 1)]
        test_target = input_target[split_index * i:split_index * (i + 1)]
        # train data
        train_data = np.concatenate((input_data[:split_index * i], input_data[split_index * (i + 1):]), axis=0)
        train_target = np.concatenate((input_target[:split_index * i], input_target[split_index * (i + 1):]), axis=0)
        # training
        training_results = training(train_data, train_target, learn_rate,
                                    hidden_perceptrons, max_epochs,
                                    early_stop_bool, act_func, early_stop_patience)
        # testing
        test_results = feed_forward(test_data, test_target,
                                    training_results[1][0], training_results[1][1],
                                    hidden_perceptrons, output_perceptrons, act_func)

        start_weights_hidden_list.append(training_results[0][0])
        start_weights_out_list.append(training_results[0][1])
        final_weights_hidden_list.append(training_results[1][0])
        final_weights_out_list.append(training_results[1][0])
        epoch_counters.append(training_results[2])
        validation_losses.append(training_results[3])
        output_hidden.append(test_results[0])
        output_final.append(test_results[1])
        all_test_targets.extend(test_target)
        all_test_predictions.extend(test_results[1])

    end_time = time.time()
    stopped_early = True if np.mean(epoch_counters) == max_epochs else False
    avg_training_results = [[np.mean(start_weights_hidden_list, axis=0), np.mean(start_weights_out_list, axis=0)],
                            [np.mean(final_weights_hidden_list, axis=0), np.mean(final_weights_out_list, axis=0)],
                            np.mean(epoch_counters),
                            np.mean(validation_losses)]
    all_test_predictions = np.array(all_test_predictions)
    all_test_targets = np.array(all_test_targets)
    test_results = [all_test_targets, all_test_predictions]
    all_info = dict_resume(hidden_perceptrons, output_perceptrons,
                           learn_rate, act_func, early_stop_bool, early_stop_patience,
                           stopped_early, threshold,
                           cross_validation_bool, cross_validation_folds,
                           avg_training_results, test_results, output_loc,
                           start_time, end_time, max_epochs)
    confusion_matrix = confusion_matrix(all_test_targets, all_test_predictions, output_perceptrons, threshold)
    markdown_report(f"{output_loc}/{output_file}", all_info, confusion_matrix)

else:  # simple holdout
    # Training data and testing data, +- 80/20
    split_index = int(alphabet_repetition * 0.8) * output_perceptrons
    # test data
    test_target = input_target[split_index:]
    test_data = input_data[split_index:]
    # train data
    train_data = input_data[:split_index]
    train_target = input_target[:split_index]
    # training
    training_results = training(train_data, train_target,
                                learn_rate, hidden_perceptrons, max_epochs,
                                early_stop_bool, act_func, early_stop_patience)
    end_time = time.time()
    # testing
    testing_results = feed_forward(test_data, test_target,
                                   training_results[1][0], training_results[1][1],
                                   hidden_perceptrons, output_perceptrons, act_func)
    all_info = dict_resume(hidden_perceptrons, output_perceptrons,
                           learn_rate, act_func, early_stop_bool, early_stop_patience,
                           training_results[4], threshold,
                           cross_validation_bool, cross_validation_folds,
                           training_results, testing_results, output_loc,
                           start_time, end_time, max_epochs)
    confusion_matrix = confusion_matrix(test_target, testing_results[1], output_perceptrons, threshold)
    markdown_report(f"{output_loc}/{output_file}", all_info, confusion_matrix)
