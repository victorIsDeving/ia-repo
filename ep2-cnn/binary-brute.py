# VICTOR AUGUSTO COSTA MONTEIRO
# NUSP 8942937
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import Callback


# Callback para registrar a perda em cada época
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# Local para salvar os arquivos de resultado
output_loc = './results-binary-brute/'
input_shape = (28, 28, 1)
target = 5
epochs, batches = 10, 200
kernel_size, pool_size = (3, 3), (2, 2)
conv_layer1_filter, conv_layer1_activation = 32, 'relu'
conv_layer2_filter, conv_layer2_activation = 64, 'relu'
dense_layer1_units, dense_layer1_activation = 128, 'relu'
output_layer_units, output_layer_activation = 1, 'sigmoid'
optimizer, loss, metrics = 'adam', 'binary_crossentropy', ['accuracy']

# Carregar os dados MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Ajustar dados
start_time_data = time.time()
x_train = x_train.reshape(x_train.shape[0], input_shape[0], input_shape[1], input_shape[2])
x_test = x_test.reshape(x_test.shape[0], input_shape[0], input_shape[1], input_shape[2])
x_train = x_train.astype('float32') / 255  # Normalizar
x_test = x_test.astype('float32') / 255
y_train = (y_train == target).astype(int)  # Binário
y_test = (y_test == target).astype(int)
end_time_data = time.time()

hyperparameters = {
    'data': 'brute data',
    'classification': f'binary, target = {target}',
    'input_shape': input_shape,
    'conv_layers': [
        {'filters': conv_layer1_filter, 'kernel_size': kernel_size, 'activation': conv_layer1_activation, 'pool_size': pool_size},
        {'filters': conv_layer2_filter, 'kernel_size': kernel_size, 'activation': conv_layer2_activation, 'pool_size': pool_size}
    ],
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

# Salvar hiper parâmetros como JSON
with open(f'{output_loc}hyperparameters.json', 'w') as json_file:
    json.dump(hyperparameters, json_file, indent=4)

# Construir o modelo
model = Sequential()
# Primeira camada convolucional
model.add(Conv2D(conv_layer1_filter, kernel_size=kernel_size, activation=conv_layer1_activation, input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size))
# Segunda camada convolucional
model.add(Conv2D(conv_layer2_filter, kernel_size=kernel_size, activation=conv_layer2_activation))
model.add(MaxPooling2D(pool_size=pool_size))
# Camada de flattening
model.add(Flatten())
# Camada densa totalmente conectada
model.add(Dense(dense_layer1_units, activation=dense_layer1_activation))
# Camada de saída
model.add(Dense(output_layer_units, activation=output_layer_activation))

# Pesos iniciais
initial_weights = model.get_weights()
np.savez(f'{output_loc}initial_weights', *initial_weights)

# Compilar
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Histórico de perda
history = LossHistory()

# Treinar o modelo
start_time_train = time.time()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batches, verbose=2, callbacks=[history])
end_time_train = time.time()

# Pesos finais
final_weights = model.get_weights()
np.savez(f'{output_loc}final_weights', *final_weights)

# Erros cometidos pela rede neural em cada iteração
np.savez(f'{output_loc}training_losses', *history.losses)

# Teste
scores = model.evaluate(x_test, y_test, verbose=0)
accuracy = scores[1] * 100
print(f"Test accuracy: {accuracy:.2f}%")

# Previsões
predictions = model.predict(x_test)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Salvar saídas produzidas pela rede neural para cada um dos dados de teste
np.savez(f'{output_loc}test_predictions', *predictions)

# Plotar 10 previsões
for i in range(10):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Prediction: {predicted_classes[i]}")
    plt.savefig(f'{output_loc}predictions_examples/prediction_{i}.png')
    plt.show()

# Matriz de confusão
true_classes = y_test
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Plotar a matriz de confusão
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not 1', '1'], yticklabels=['Not 1', '1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(f'{output_loc}confusion_matrix.png')
plt.show()

data_time = end_time_data - start_time_data
print(f'Hog extraction time: {data_time}')
train_time = end_time_train - start_time_train
print(f'Training time: {train_time}')

infos = {
    'Data prep time': f'{data_time}',
    'Training time': f'{train_time}',
    'Total time': f'{data_time + train_time}',
    'Accuracy': f'{accuracy}'
}

with open(f'{output_loc}infos.json', 'w') as json_file:
    json.dump(infos, json_file, indent=4)
