""" """
import math
import numpy as np

def sigmoid(x: float or int) -> float:
    """ Returns the sigmoid to the given x
    """

    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x: float):
    """ Returns the derivative of a
        given sigmoid result
    """
    
    return x * (1 - x)


class Node:

    def __init__(self):
        self.nodes
        self.layers = list()


class Network:

    def __init__(self):
        self.layers = list()

    def feedforward(self):  
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2