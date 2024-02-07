import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros(output_size)

    def forward(self, inputs):
        self.original_shape = inputs.shape
        
        flat_inputs = inputs.flatten()
        self.inputs = flat_inputs

        self.output = np.dot(flat_inputs, self.weights) + self.biases

        return self.output

    def backprop(self, gradient, learning_rate):
        weight_gradient = np.outer(self.inputs, gradient)
        bias_gradient = gradient

        self.weights -= learning_rate * weight_gradient.reshape(self.weights.shape)
        self.biases -= learning_rate * bias_gradient

        return np.dot(gradient, self.weights.T).reshape(self.original_shape)
