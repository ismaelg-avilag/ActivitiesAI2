"""
Actividad: A05. Red Neuronal de una Capa
Inteligencia Artificial 2

Clase ONL
"""

import numpy as np
from activationFunctions import *


class OLN:
    def __init__(self, n_inputs, n_outputs, activation_function = linear):
        self.w = -1 + 2 * np.random.rand(n_outputs, n_inputs)
        self.b = -1 + 2 * np.random.rand(n_outputs, 1)
        self.f = activation_function
    
    def predict(self, X):
        Z = self.w @ X + self.b
        return self.f(Z)
    
    def fit(self, X, Y, epochs = 500, lr = 0.1): # lr = learning_rate
        p = X.shape[1]
        
        # batch version
        for _ in range(epochs):
            # propagation
            Z = self.w @ X + b
            Y_est, dY = self.f(z, derivative = True)
            
            # calculate the local gradient
            lg = (Y - Y_est) * dY
            
            # parameter update
            self.w += (lr / p) * lg @ X.T
            self.b += (lr / p) * np.sum(lg)