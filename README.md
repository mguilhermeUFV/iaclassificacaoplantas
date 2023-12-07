# Relatório do Projeto de Classificacação de doenças em folhas de diferentes culturas agrícolas

# Repositório de Treinamento de IA com CNN Sequencial

Este repositório contém o código utilizado no treinamento de uma Rede Neural Convolucional (CNN) sequencial para o projeto da disciplina de Inteligência Artificial da Universidade Federal de Viçosa - Campus de Rio Paranaíba.

## Descrição

Neste projeto, desenvolvemos uma IA capaz de realizar a classificação de doenças em plantas usando uma abordagem de CNN sequencial. Utilizamos um conjunto de dados de imagens de plantas afetadas por doenças para treinar e avaliar o modelo.

## Estrutura do Repositório

- O notebook Jupyter contendo o código-fonte, treinamento do modelo e análise de resultados.
- [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset): O conjunto de dados de imagens de plantas com doenças.

## Configuração do Ambiente

Para executar este projeto, é necessário configurar um ambiente com as seguintes bibliotecas:

- TensorFlow
- Matplotlib
- Scikit-learn
- Google Colab

Você pode montar o Google Drive para acessar os dados e executar o código no ambiente Colab, conforme demonstrado no notebook.

## Configurando o Ambiente

```python
# Montar o Google Drive para acessar os dados
from google.colab import drive
drive.mount('/content/drive')

# Descompactar o arquivo de dados
!unzip '/content/drive/My Drive/PlantDiseases.zip' -d '/content/drive/My Drive/'

# Importar bibliotecas e configurar avisos
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(0)
from tensorflow import keras
import numpy as np
np.random.seed(0)
import itertools
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
````

## Carregamento e Pré-processamento dos Dados
```python
# Carregar conjuntos de dados de treinamento e teste
gerador_treino = image_dataset_from_directory(directory="/content/drive/My Drive/New Plant Diseases Dataset(Augmented)/train",
                                              image_size=(256, 256))
gerador_teste = image_dataset_from_directory(directory="/content/drive/My Drive/New Plant Diseases Dataset(Augmented)/valid",
                                             image_size=(256, 256))

# Reescalar as imagens
reescala = Rescaling(scale=1.0/255)
gerador_treino = gerador_treino.map(lambda imagem, etiqueta: (reescala(imagem), etiqueta))
gerador_teste = gerador_teste.map(lambda imagem, etiqueta: (reescala(imagem), etiqueta))
```

## Modelo de Rede Neural
```python
modelo = keras.Sequential()

modelo.add(keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)))
modelo.add(keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
modelo.add(keras.layers.MaxPooling2D(3, 3))

modelo.add(keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
modelo.add(keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
modelo.add(keras.layers.MaxPooling2D(3, 3))

modelo.add(keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
modelo.add(keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
modelo.add(keras.layers.MaxPooling2D(3, 3))

modelo.add(keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
modelo.add(keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"))

modelo.add(keras.layers.Conv2D(512, (5, 5), activation="relu", padding="same"))
modelo.add(keras.layers.Conv2D(512, (5, 5), activation="relu", padding="same"))

modelo.add(keras.layers.Flatten())

modelo.add(keras.layers.Dense(1500, activation="relu"))
modelo.add(keras.layers.Dropout(0.5))

modelo.add(keras.layers.Dense(38, activation="softmax"))

otimizador = keras.optimizers.Adam(learning_rate=0.0001)
modelo.compile(optimizer=otimizador, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
modelo.summary()
```
## Treinamento do modelo
```python
epocas = 8
historico = modelo.fit(gerador_treino,
                       validation_data=gerador_teste,
                       epochs=epocas)
```

## Avaliação do Modelo
```python
plt.figure(figsize=(20, 5))

# Gráfico de Perda (Loss)
plt.subplot(1, 2, 1)
plt.title("Perda de Treino e Validação")
plt.xlabel("Época")
plt.ylabel("Perda")
plt.plot(historico.history['loss'], label="Perda de Treino")
plt.plot(historico.history['val_loss'], label="Perda de Validação")
plt.xlim(0, 10)
plt.ylim(0.0, 1.0)
plt.legend()

# Gráfico de Precisão (Accuracy)
plt.subplot(1, 2, 2)
plt.title("Acuracia de Treino e Validação")
plt.xlabel("Época")
plt.ylabel("Acuracia")
plt.plot(historico.history['accuracy'], label="Acuracia de Treino")
plt.plot(historico.history['val_accuracy'], label="Acuracia de Validação")
plt.xlim(0, 9.25)
plt.ylim(0.75, 1.0)
plt.legend()

plt.tight_layout()

rotulos = []
previsoes = []

for x, y in gerador_teste:
    rotulos.append(list(y.numpy()))
    previsoes.append(tf.argmax(modelo.predict(x), 1).numpy())

previsoes = list(itertools.chain.from_iterable(previsoes))
rotulos = list(itertools.chain.from_iterable(rotulos))

print("Acuracia do Treino: {:.2f} %".format(historico.history['accuracy'][-1] * 100))
print("Acuracia do Teste : {:.2f} %".format(accuracy_score(rotulos, previsoes) * 100))
print("Precisão : {:.2f} %".format(precision_score(rotulos, previsoes, average='micro') * 100))
print("Recall : {:.2f} %".format(recall_score(rotulos, previsoes, average='micro') * 100))
```
## Matriz de Confusão
```python
plt.figure(figsize=(20, 5))
cm = confusion_matrix(rotulos, previsoes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(1, 39)))
fig, ax = plt.subplots(figsize=(15, 15))
disp.plot(ax=ax, colorbar=False, cmap='YlGnBu')
plt.title("Matriz de Confusão")
plt.xlabel('Rótulos Previstos')
plt.ylabel('Rótulos Verdadeiros')
plt.show()
```
