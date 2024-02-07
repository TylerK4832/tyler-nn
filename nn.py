import numpy as np

class NN:
    def __init__(self, input_size, layers, activation_functions, loss_function, learning_rate=0.01):

        self.weights = []
        self.biases = []

        # Initialization of weights and biases
        d1 = input_size
        for d2 in layers:
            # self.weights.append(np.random.randn(d2, d1)*np.sqrt(2.0/d1))
            self.weights.append(np.random.randn(d2, d1))
            self.biases.append(np.zeros((d2,1)))
            d1 = d2

        # Initialize weights and biases with random values
        # self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        # self.bias_hidden = np.zeros((1, hidden_size))
        # self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        # self.bias_output = np.zeros((1, output_size))

        self.activation_functions = activation_functions
        self.loss_function = loss_function
        self.learning_rate = learning_rate
    
    def forward(self, input_data):
        self.layer_outputs = [np.array(input_data)]

        for i in range(len(self.weights)):
            layer_input = np.dot(self.layer_outputs[-1], self.weights[i].T) + self.biases[i]
            layer_output = self.activation_functions[i].activate(layer_input)
            self.layer_outputs.append(layer_output)

        return self.layer_outputs[-1]

    def backward(self, input_data, target):
        # Calculate the gradient of the loss with respect to the output
        output_error = self.loss_function.lossGradient(target, self.layer_outputs[-1])

        # Backpropagate the error through each layer
        for i in range(len(self.weights) - 1, -1, -1):
            current_output = self.layer_outputs[i + 1]
            print("\nCurrent output:\n", str(current_output))
            current_input = np.dot(current_output, self.weights[i]) + self.biases[i]
            activation_derivative = self.activation_functions[i].backprop_grad(current_input)

            output_error = output_error * activation_derivative
            d_weights = np.dot(self.layer_outputs[i].T, output_error)
            d_biases = np.sum(output_error, axis=1, keepdims=True)

            # Update weights and biases using gradient descent
            print("\ncurrent weight:\n", str(self.weights[i]))
            print("\nd_weight:\n", str(d_weights))
            self.weights[i] -= self.learning_rate * d_weights
            self.biases[i] -= self.learning_rate * d_biases

            # Calculate error for the next layer
            output_error = np.dot(output_error, self.weights[i].T)

    def train(self, input_data, target, epochs=1):
        for epoch in range(epochs):
            for i in range(len(input_data)):
                # Forward pass
                output = self.forward(input_data[i])
                print("Weights:")
                print(self.weights)
                print("\nBiases:")
                print(self.biases)
                print("\nLayer outputs:")
                print(self.layer_outputs)

                # Backward pass
                self.backward(input_data[i], np.array(target[i]))

    def predict(self, input_data):
        probabilities = self.forward(input_data)[0]
        max_index = np.argmax(probabilities)
        output = np.zeros_like(probabilities)
        output[max_index] = 1
        output = output.astype(int)
        return output
    

class activationFunction:

    def activate(self,X):
        raise NotImplementedError("Abstract class.")

    def backprop_grad(self, grad):
        raise NotImplementedError("Abstract class.")

class Relu(activationFunction):
    def activate(self,X):
        return X*(X>0)

    def backprop_grad(self, X):
        return (X>0).astype(np.float64)

class Linear(activationFunction):
    def activate(self,X):
        return X
    
    def backprop_grad(self,X):
        return np.ones(X.shape, dtype=np.float64)
    
class Sigmoid(activationFunction):
    def activate(self, X):
        return 1 / (1 + np.exp(-X))

    def backprop_grad(self, grad):
        return grad * (1 - grad)
    
class Softmax(activationFunction):
    def activate(self, X):
        exp_values = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    def backprop_grad(self, grad):
        return grad


class LossFunction:
    def loss(self, Y, Yhat):
        raise NotImplementedError("Abstract class.")

    def lossGradient(self, Y, Yhat):
        raise NotImplementedError("Abstract class.")

class SquaredLoss(LossFunction):
    def loss(self, Y, Yhat):
        return 0.5 * np.sum(np.square(Yhat - Y))

    def lossGradient(self, Y, Yhat):
        return Yhat - Y
    
class HingeLoss(LossFunction):
    def loss(self, Y, Yhat):
        return np.maximum(0, 1 - Y * Yhat).mean()

    def lossGradient(self, Y, Yhat):
        return np.where(Y * Yhat < 1, -Y, 0)