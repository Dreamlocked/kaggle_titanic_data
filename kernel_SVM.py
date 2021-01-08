# -*- coding: utf-8 -*-
"""
@author: Dreamlocked
"""


# Importando las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('train.csv')

# Eliminando NaN
# dataset[11] = dataset[11].replace(np.nan,0)
dataset['Cabin'] = dataset['Cabin'].replace(np.nan, 0)
for i in range(0,len(dataset['Cabin'])):
    if type(dataset['Cabin'][i]) is str:
        dataset['Cabin'][i] = 1
dataset['Age'] = dataset['Age'].replace(np.nan, int(dataset['Age'].mean()))
dataset['Embarked'] = dataset['Embarked'].replace(np.nan, "S")

X = dataset.iloc[:, [2,4,5,6,7,9,10,11]].values
y = dataset.iloc[:, 1].values



# Codificar datos categoricos genero
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

# Codificar datos categoricos boleto
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough')

X = np.array(ct.fit_transform(X))
# Forma bruta de eliminar una columna, aunque no se elimina de verdad
X = X[:,1:]

# Codificar datos de embarque
labelencoder_X = LabelEncoder()
X[:, 8] = labelencoder_X.fit_transform(X[:, 8])

ct_1 = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [8])],   
    remainder='passthrough')
X = np.array(ct_1.fit_transform(X))
X = X[:,1:]

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# from sklearn.decomposition import KernelPCA
# kpca = KernelPCA(n_components = 6, kernel = "rbf")
# X_train = kpca.fit_transform(X_train)
# X_test = kpca.transform(X_test)

# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel = "rbf", random_state = 0)
classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 20)
accuracies.mean()
accuracies.std()

# Ajustar el modelo XGBoost al Conjunto de Entrenamiento
from xgboost import XGBClassifier
classifier_two = XGBClassifier()
classifier_two.fit(X_train, y_train)

y_pred_two  = classifier_two.predict(X_test)
cm_two = confusion_matrix(y_test, y_pred_two)

accuracies_two = cross_val_score(estimator = classifier_two, X = X_train, y = y_train, cv = 20)
accuracies_two.mean()
accuracies_two.std()