import os
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix


def load_images(image_dir, cnn=False, target_size=(100, 100)):
    """ Carrega as imagens
    :param image_dir: diretório onde se encontram as duas pastas 'ok_front' e 'def_front'
                        com as imagens normais e com defeitos, respectivamente.
    :param cnn: se será uitlizado ou não a cnn para extração de atributos
    :param target_size: reshape da imagem caso não esteja utilizando a cnn para extração de atributos
    :return:
    """
    images = []
    labels = []
    image_ids = []

    for label_dir in ['def_front', 'ok_front']:
        class_dir = os.path.join(image_dir, label_dir)
        label = 1 if label_dir == 'def_front' else 0

        for i, file_name in enumerate(os.listdir(class_dir)):
            image = Image.open(os.path.join(class_dir, file_name))
            if not cnn:
                image = image.convert("L")
                image = image.resize(target_size)
            image = np.array(image)
            images.append(image)
            labels.append(label)
            image_ids.append(file_name)
            if i % 1000 == 0:
                print(f"Carregadas {i} imagens de {class_dir}")

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, image_ids


def feature_extraction(images, features='lbp', P=15, R=5):
    """ Extração de atributos das imagens
    :param images: lista de imagens
    :param features: que tipo de extrator utilizar: 'lbp' ou 'hog' ou 'flatten'
    :param P: número de pontos ao redor de cada pixel que serão considerados
    :param R: raio da vizinhança circular ao redor de cada pixel
    :return: lista de atributos
    """
    preprocessed_images = []
    images = images.astype(np.float64)

    for idx, image in enumerate(images):
        image = image / 255.0  # Normalizar a imagem para o intervalo [0, 1]

        if features == 'hog':
            image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        elif features == 'lbp':
            # Técnica utilizada para análise de texturas
            lbp = local_binary_pattern(image, P, R, method="uniform")  # calcula o padrão local binário (LBP) da imagem
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))  #  calcula o histograma dos valores do LBP
            hist = hist.astype("float")
            image /= (hist.sum() + 1e-6)  # número total de pixels na imagem considerados para os padrões LBP
            image = image.flatten()
        else:
            image = image.flatten()

        preprocessed_images.append(image)
    return np.array(preprocessed_images)


def cnn_feature_extraction(images, model):
    """ Usa uma cnn pré-treinada no imagenet para extrair atributos
    :param images: lista de imagens
    :param model: modelo a ser utilizado para extração de atributos
    :return: lista de atributos
    """
    preprocessed_images = np.array([preprocess_input(img_to_array(Image.fromarray(image).resize((224, 224)))) for image in images])
    features = model.predict(preprocessed_images, batch_size=32)
    return features.reshape(features.shape[0], -1)


def plot_results(y_test, inferences):
    """Imprime o report de classificação e a matriz de confusão

    :param y_test: os labels reais
    :param inferences: dicionário com inferências de n modelos. Cada modelo é um par chave:valor,
                        onde a chave é a string com nome do modelo e a chave é o array de inferências.
    """
    plt.figure(figsize=(14, 6))
    for idx, (key, value) in enumerate(inferences.items()):
        print("Report de Classificação para", key)
        print(classification_report(y_test, value))

        cm = confusion_matrix(y_test, value, normalize='true')
        plt.subplot(1, len(inferences), idx+1)
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                    xticklabels=['Normal', 'Anomalia'], yticklabels=['Normal', 'Anomalia'])
        plt.title('Matriz de Confusão - ' + key)
        plt.xlabel('Previsto')
        plt.ylabel('Real')
    plt.show()
