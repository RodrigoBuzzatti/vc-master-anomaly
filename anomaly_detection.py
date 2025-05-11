from utils import *

import numpy as np
from keras.models import Model
from sklearn.svm import OneClassSVM
from keras.applications import VGG16
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier

# Parâmetros
use_cnn = True
features = 'lbp'  # 'lbp' ou 'hog' ou 'flatten'

train_dir = r'D:\rodri\Python-Code\vc-master-anomaly\data\train'
test_dir = r'D:\rodri\Python-Code\vc-master-anomaly\data\test'

# Carregar as imagens e metadados de treino
train_images, train_labels, train_image_ids = load_images(train_dir, cnn=use_cnn)
test_images, test_labels, test_image_ids = load_images(test_dir, cnn=use_cnn)

if use_cnn:
    base_model = VGG16(weights='imagenet', include_top=False)
    model_vgg = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    train_images = cnn_feature_extraction(train_images, model_vgg)
    test_images = cnn_feature_extraction(test_images, model_vgg)
else:
    train_images = feature_extraction(train_images, features=features)
    test_images = feature_extraction(test_images, features=features)

print('Shape antes do PCA:', train_images.shape)
# Aplicar PCA para reduzir a dimensionalidade
pca = PCA(n_components=0.99)  # por variância acumulada
train_images = pca.fit_transform(train_images)
test_images = pca.transform(test_images)
print('Shape após do PCA:', train_images.shape)

# Separar os dados normais (sem defeito) para o treino do OneClass SVM e do Isolation Forest
X_train = train_images[train_labels == 0]
X_test = test_images
y_test = test_labels

# Treinamento e predição dos modelos
print('Treinando Isolation Forest')
isolation_forest = IsolationForest(contamination=0.1, random_state=42)
isolation_forest.fit(X_train)

print('Treinando One-Class SVM')
one_class_svm = OneClassSVM(nu=0.2)
one_class_svm.fit(X_train)

print('Treinando Random Forest')
rf = RandomForestClassifier()
rf.fit(train_images, train_labels)

print('Realizando inferências')
y_pred_isolation_forest = isolation_forest.predict(X_test)
y_pred_one_class_svm = one_class_svm.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Conversão de labels: -1 (anomalia), 1 (normal) para 0 (normal), 1 (anomalia)
y_pred_isolation_forest = np.where(y_pred_isolation_forest == 1, 0, 1)
y_pred_one_class_svm = np.where(y_pred_one_class_svm == 1, 0, 1)

# Criar dicionário com inferências:
inferences = {'Isolation Forest': y_pred_isolation_forest,
              'OneClass SVM': y_pred_one_class_svm,
              'Random Forest': y_pred_rf}

plot_results(y_test, inferences)
