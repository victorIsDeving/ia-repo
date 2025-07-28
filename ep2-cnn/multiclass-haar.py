# VICTOR AUGUSTO COSTA MONTEIRO
# NUSP 8942937
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pywt  # BIBLIOTECA PARA A TRANSFORMADA WAVELET DE HAAR
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))

# Função para extrair características da transformada wavelet de Haar
def extract_haar_features(images):
    haar_features = []
    for image in images:
        # Realizar a transformada wavelet de Haar em 2D
        coeffs = pywt.dwt2(image, 'haar')
        cA, (cH, cV, cD) = coeffs
        # Concatenar os coeficientes em um vetor de características
        feature = np.hstack((cA.ravel(), cH.ravel(), cV.ravel(), cD.ravel()))
        haar_features.append(feature)
    return np.array(haar_features)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
output_loc = './results-multiclass-haar/'
epochs, batches = 10, 200
dense_layer1_units, dense_layer1_activation = 128, 'relu'
output_layer_units, output_layer_activation = 10, 'softmax'
optimizer, loss, metrics = 'adam', 'binary_crossentropy', ['accuracy']

(x_train, y_train), (x_test, y_test) = mnist.load_data()

start_time_haar = time.time()
x_train_haar = extract_haar_features(x_train)
x_test_haar = extract_haar_features(x_test)
x_train_haar = x_train_haar / np.max(x_train_haar)  # Normalizar
x_test_haar = x_test_haar / np.max(x_test_haar)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
end_time_haar = time.time()

hyperparameters = {
    'data': 'haar data',
    'classification': 'multiclass',
    'input_shape': (x_train_haar.shape[1],),
    'dense_layers': [
        {'units': dense_layer1_units, 'activation': dense_layer1_activation},
        {'units': output_layer_units, 'activation': output_layer_activation}
    ],
    'optimizer': optimizer,
    'loss': loss,
    'metrics': metrics,
    'epochs': epochs,
    'batch_size': batches
}

with open(f'{output_loc}hyperparameters.json', 'w') as json_file:
    json.dump(hyperparameters, json_file, indent=4)

model = Sequential()
model.add(Dense(dense_layer1_units, activation=dense_layer1_activation, input_shape=(x_train_haar.shape[1],)))
model.add(Dense(output_layer_units, activation=output_layer_activation))

initial_weights = model.get_weights()
np.savez(f'{output_loc}initial_weights.npz', *initial_weights)

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

history = LossHistory()

start_time_training = time.time()
model.fit(x_train_haar, y_train, validation_data=(x_test_haar, y_test), epochs=epochs, batch_size=batches, verbose=2, callbacks=[history])
end_time_training = time.time()

final_weights = model.get_weights()
np.savez(f'{output_loc}final_weights.npz', *final_weights)

np.savez(f'{output_loc}training_losses.npz', *history.losses)

scores = model.evaluate(x_test_haar, y_test, verbose=0)
accuracy = scores[1] * 100
print(f"Test accuracy: {accuracy:.2f}%")

predictions = model.predict(x_test_haar)
predicted_classes = np.argmax(predictions, axis=1)

np.savez(f'{output_loc}test_predictions.npz', *predictions)

for i in range(10):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Prediction: {predicted_classes[i]}")
    plt.savefig(f'{output_loc}predictions_examples/prediction_{i}.png')
    plt.close()

true_classes = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(f'{output_loc}confusion_matrix.png')
plt.show()

haar_time = end_time_haar - start_time_haar
print(f'Haar prep time: {haar_time}')
train_time = end_time_training - start_time_training
print(f'Training time: {train_time}')

infos = {
    'Data prep time': f'{haar_time}',
    'Training time': f'{train_time}',
    'Total time': f'{haar_time + train_time}',
    'Accuracy': f'{accuracy}'
}

with open(f'{output_loc}infos.json', 'w') as json_file:
    json.dump(infos, json_file, indent=4)
