"""
Actividad: A05. Red Neuronal de una Capa
Inteligencia Artificial 2

Ejemplo
"""

import pandas as pd
import matplotlib.pyplot as plt

from OLN import *

xmin, xmax = -5, 5

dataset = pd.read_csv('Dataset_A05.csv');
X = dataset.iloc[:, 0:2].values.T
Y = dataset.iloc[:, 2:].values.T

net = OLN(2, 4, logistic)
net.fit(X, Y)


# graphic
cm = [[0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
      [0, 0, 1],
      [1, 0, 1],
      [0, 1, 1],
      [1, 1, 1]]


ax1 = plt.subplot(1, 2, 1)
y_c = np.argmax(Y, axis=0)

for i in range(X.shape[1]):
    ax1.plot(X[0, i], X[1, i], '*', c=cm[y_c[i]])
ax1.axis([-5.5, 5.5, -5.5, 5.5])
ax1.grid()
ax1.set_title('Problema Original')


ax2 = plt.subplot(1, 2, 2)
y_c = np.argmax(net.predict(X), axis=0)

for i in range(X.shape[1]):
    ax2.plot(X[0, i], X[1, i], '*', c=cm[y_c[i]])
ax2.axis([-5.5, 5.5, -5.5, 5.5])
ax2.grid()
ax2.set_title('Predicción de la red')

plt.show()
