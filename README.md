# Relatório do Projeto Iaplantas

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
