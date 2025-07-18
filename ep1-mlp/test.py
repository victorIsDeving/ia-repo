import numpy as np


def confusion_matrix(y_true, y_pred, num_classes):
    # Inicializar a matriz de confusão com zeros
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Preencher a matriz de confusão
    for true_label, pred_label in zip(y_true, y_pred):
        conf_matrix[true_label, pred_label] += 1

    return conf_matrix


# Exemplo de uso:
y_true = np.array([0, 1, 2, 1, 0, 2, 2, 1, 0, 2])
y_pred = np.array([0, 2, 1, 1, 0, 2, 2, 0, 0, 2])

num_classes = 3
conf_matrix = confusion_matrix(y_true, y_pred, num_classes)

print("Confusion Matrix:")
print(conf_matrix)
