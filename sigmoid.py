import numpy as np

class Sigmoid:
    def __init__(self):
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        return 1 / (1 + np.exp(-np.clip(input_data, -500, 500)))

    def backprop(self, gradient, lr):
        sigmoid_output = self.forward(self.input)
        sigmoid_derivative = sigmoid_output * (1 - sigmoid_output)
        return gradient * sigmoid_derivative
