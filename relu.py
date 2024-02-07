import numpy as np

class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backprop(self, gradient, lr):
        return gradient * (self.input > 0)
