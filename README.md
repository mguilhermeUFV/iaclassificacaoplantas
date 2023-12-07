# Relatório do Projeto Iaplantas

## Introdução

Este é um relatório gerado automaticamente pelo Colaboratory. O código e os resultados aqui apresentados referem-se ao projeto Iaplantas, que visa a classificação de doenças de plantas.

O arquivo original pode ser encontrado [aqui](https://colab.research.google.com/drive/1FskVVhxBvrMqSJ7rJH3_hXNCPkVqbtWG).

## Configuração do Ambiente

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
