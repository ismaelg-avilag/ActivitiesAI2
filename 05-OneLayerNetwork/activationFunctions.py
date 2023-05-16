"""
Actividad: A05. Red Neuronal de una Capa
Inteligencia Artificial 2

Funciones de Activación para la OLN
"""

import numpy as np


# Activation Functions

def linear(z, derivative = False): # regresion
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

def logistic(z, derivative = False): # multi-label
    a = 1 / (1 + np.exp(-z))
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

def softmax(z, derivative = False): # classification with a single winner
    e_z = np.exp(z - np.max(z, axis=0))
    a = e_z / np.sum(e_z, axis=0)
    
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a