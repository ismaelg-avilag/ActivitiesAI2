"""
Actividad: A02. El Algoritmo del Perceptron
Inteligencia Artificial 2
"""

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, n_input, learning_rate):
        self.w = -1 + 2 * np.random.rand(n_input)
        self.b = -1 + 2 * np.random.rand()
        self.eta = learning_rate


    def predict(self, X):
        _, p = X.shape
        y_est = np.zeros(p)

        for i in range(p):
            y_est[i] = np.dot(self.w, X[:, i]) + self.b
            if(y_est[i]) >= 0:
                y_est[i] = 1
            else:
                y_est[i] = 0

        return y_est


    def fit(self, X, Y, epochs = 50):
        _, p = X.shape
        for _ in range(epochs):
            for i in range(p):
                y_est = self.predict(X[:, i].reshape(-1, 1))
                self.w += self.eta * (Y[i] - y_est) * X[:, i]
                self.b += self.eta * (Y[i] - y_est)


# ***************Ejemplo***************
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y = np.array([0, 1, 1, 1])

model = Perceptron(2, 0.1)
print(model.predict(X))

model.fit(X, Y)
print(model.predict(X))


# ***************Grafica***************
def draw_2d_perceptron(net):
    w1, w2, b = net.w[0], net.w[1], net.b
    plt.plot([-2, 2], [(1/w2) * (-w1 * (-2)-b), (1/w2) * (-w1 * 2 -b), '--k'])
    
_, p = X.shape
for i in range(p):
    if Y[i] == 0:
        plt.plot(X[0, i], X[1, i], 'or') #punto rojo
    else:
        plt.plot(X[0, i], X[1, i], 'ob') #punto azul
        
plt.title('Perceptron')
plt.grid('on')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

draw_2d_perceptron(model)