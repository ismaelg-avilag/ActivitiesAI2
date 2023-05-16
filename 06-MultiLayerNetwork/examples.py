"""
Actividad: A06. Red Neruonal Multicapa
Inteligencia Artificial 2

Ejemplos
"""

import pandas as pd

from graphics import *
from MLP import *


def blobs():
    dataset = pd.read_csv('blobs.csv')
    X = dataset.iloc[:, 0:2].values.T
    Y = dataset.iloc[:, 2:].values.T
    
    net = DenseNetwork((2, 20, 1))
    print(net.predict(X))
    MLP_binary_classification_2d(X, Y, net)
    
    net.fit(X, Y)
    print(net.predict(X))
    MLP_binary_classification_2d(X, Y, net)


def circles():
    dataset = pd.read_csv('circles.csv')
    X = dataset.iloc[:, 0:2].values.T
    Y = dataset.iloc[:, 2:].values.T
    
    net = DenseNetwork((2, 20, 1))
    print(net.predict(X))
    MLP_binary_classification_2d(X, Y, net)
    
    net.fit(X, Y)
    print(net.predict(X))
    MLP_binary_classification_2d(X, Y, net)


def moons():
    dataset = pd.read_csv('moons.csv')
    X = dataset.iloc[:, 0:2].values.T
    Y = dataset.iloc[:, 2:].values.T
    
    net = DenseNetwork((2, 20, 1))
    print(net.predict(X))
    MLP_binary_classification_2d(X, Y, net)
    
    net.fit(X, Y)
    print(net.predict(X))
    MLP_binary_classification_2d(X, Y, net)


def xor():
    dataset = pd.read_csv('XOR.csv')
    X = dataset.iloc[:, 0:2].values.T
    Y = dataset.iloc[:, 2:].values.T
    
    net = DenseNetwork((2, 20, 1))
    print(net.predict(X))
    MLP_binary_classification_2d(X, Y, net)
    
    net.fit(X, Y)
    print(net.predict(X))
    MLP_binary_classification_2d(X, Y, net)