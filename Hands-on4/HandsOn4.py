# -*- coding: utf-8 -*-
"""
Created on Sat May 18 22:32:52 2024

@author: MIGUEL HDZ
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

"""
En esta actividad, implementé un clasificador KNN (k vecinos más cercanos) y lo visualizo
junto con los límites de decisión sobre un conjunto de datos bidimensional. Ademas defino
una clase KNNVisualizer que encapsula la funcionalidad de ajustar el modelo KNN, para 
graficar los límites de decisión y mostrar las predicciones.
Utilizo numpy para manipular matrices, matplotlib.pyplot para graficar y sklearn.neighbors.KNeighborsClassifier
para entrenar el modelo KNN.

La función main genera un objeto KNNVisualizer, ajusta el modelo a los datos de ejemplo,
que realiza predicciones, grafica los límites de decisión y muestra los resultados.
"""

class KNNVisualizer:
    def __init__(self, k):
        self.k = k
        self.knn = None

    def fit(self, X, y):
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.knn.fit(X, y)

    def plot_decision_boundary(self, X, y):
        h = 0.02  # Paso de la malla
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.knn.predict(np.c_[xx.ravel(), yy.ravel()])

        # Crear un mapa de colores para las clases
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

        # Graficar los límites de decisión
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Graficar los puntos de datos
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("KNN con k = {}".format(self.k))
        plt.xlabel("Característica 1")
        plt.ylabel("Característica 2")
        plt.show()

    def print_results(self, X_test, y_test, y_pred):
        print("Predicciones del modelo:")
        for i in range(len(X_test)):
            print("Características:", X_test[i], " | Clase real:", y_test[i], " | Clase predicha:", y_pred[i])

def main():
    # Datos de ejemplo
    X = np.array([[2, 3], [5, 7], [8, 9], [1, 4], [6, 2], [3, 8], [7, 5], [4, 1], [9, 6], [2, 8],
                  [7, 2], [5, 9], [3, 6], [8, 3], [1, 7], [6, 4], [4, 9], [9, 1], [2, 5], [7, 8]])
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1])

    # Entrenar el clasificador KNN
    knn_visualizer = KNNVisualizer(k=3)
    knn_visualizer.fit(X, y)

    # Realizar predicciones en el conjunto de datos de ejemplo
    y_pred = knn_visualizer.knn.predict(X)

    # Graficar los límites de decisión y los puntos de datos
    knn_visualizer.plot_decision_boundary(X, y)

    # Imprimir resultados
    knn_visualizer.print_results(X, y, y_pred)

if __name__ == "__main__":
    main()
