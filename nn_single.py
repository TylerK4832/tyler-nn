import numpy as np

class NN:
    def __init__(self, input_size, hidden_size, output_size,learning_rate=0.01):
        # Initialize weights and biases with random values
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

        # Learning rate for gradient descent
        self.learning_rate = learning_rate

    def forward(self, input_data):
        # Input to hidden layer
        self.hidden_input = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
        sig = Sigmoid()
        self.hidden_output = sig.activate(self.hidden_input)

        # Hidden to output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        soft = Softmax()
        self.output = soft.activate(self.output_input)

        return self.output

    def backward(self, input_data, target):
        # Calculate the gradient of the loss with respect to the output
        output_error = self.output - target

        # Calculate gradients for the weights and biases in the output layer
        d_weights_hidden_output = np.dot(self.hidden_output.T, output_error)
        d_bias_output = np.sum(output_error, axis=0, keepdims=True)

        # Backpropagate the error to the hidden layer
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * (self.hidden_output * (1 - self.hidden_output))

        # Calculate gradients for the weights and biases in the hidden layer
        d_weights_input_hidden = np.dot(input_data.T, hidden_error)
        d_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True)

        # Update weights and biases using gradient descent
        self.weights_hidden_output -= self.learning_rate * d_weights_hidden_output
        self.bias_output -= self.learning_rate * d_bias_output
        self.weights_input_hidden -= self.learning_rate * d_weights_input_hidden
        self.bias_hidden -= self.learning_rate * d_bias_hidden

    def train(self, input_data, target, epochs=1):
        for i in range(epochs):
            output = self.forward(input_data)
            self.backward(input_data, target)

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
        return (1.0 / (2 * Y.shape[1])) * (np.linalg.norm(Yhat - Y) ** 2)

    def lossGradient(self, Y, Yhat):
        return (1.0 / Y.shape[1]) * (Yhat - Y)
    
class HingeLoss(LossFunction):
    def loss(self, Y, Yhat):
        return np.maximum(0, 1 - Y * Yhat).mean()

    def lossGradient(self, Y, Yhat):
        return np.where(Y * Yhat < 1, -Y, 0)