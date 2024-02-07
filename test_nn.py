import numpy as np
import tensorflow as tf

class NeuralNetwork:
    def __init__(self, layers, activation_functions, loss_function):
        self.layers = layers

        self.weights = [np.random.randn(layers[i], layers[i+1])*np.sqrt(2.0/layers[i]) for i in range(len(layers)-1)]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]
        self.activations = [np.zeros((1, layer)) for layer in layers]

        self.activation_functions = activation_functions
        self.loss_function = loss_function

    def forward(self, input_data):
        self.activations[0] = input_data
        # print("\nstarting forward pass:")
        # print(self.activations[0])
        for i in range(len(self.layers)-2):
            self.activations[i+1] = self.activation_functions[i].activate(np.dot(self.activations[i], self.weights[i]) + self.biases[i])
            # print(self.activations[i+1])
        output = self.activation_functions[-1].activate(np.dot(self.activations[-2], self.weights[-1]) + self.biases[-1])
        # output = self.activations[-1]
        # print("output:")
        # print(output)
        return output

    def backward(self, output, target, learning_rate):
        error = self.loss_function.lossGradient(target, output)
        delta = error * self.activation_functions[-1].backprop_grad(output)
        # print("delta: \n", str(delta))
        self.weights[-1] -= np.dot(self.activations[-2].T, delta)
        self.biases[-1] -= np.sum(delta, axis=0, keepdims=True)
        for i in range(len(self.layers)-3, -1, -1):
            delta = np.dot(delta, self.weights[i+1].T) * self.activation_functions[i+1].backprop_grad(self.activations[i+1])
            self.weights[i] -= learning_rate * np.dot(self.activations[i].T, delta)
            self.biases[i] -= learning_rate * np.sum(delta, axis=0, keepdims=True)

    def train(self, input_data, target, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(input_data)
            # print("target: \n", str(target))
            # print("output: \n", str(output))
            self.backward(output, target, learning_rate)
            loss = self.loss_function.loss(target, output)
            print(f"Epoch {epoch}, Loss: {loss}")

    def train_batch(self, input_data, target, epochs, learning_rate, batch_size):
        num_samples = len(input_data)

        for epoch in range(epochs):
            # Shuffle the data for each epoch
            indices = np.random.permutation(num_samples)
            input_data_shuffled = input_data[indices]
            target_shuffled = target[indices]

            # Process the data in batches
            for i in range(0, num_samples, batch_size):
                batch_input = input_data_shuffled[i:i+batch_size]
                batch_target = target_shuffled[i:i+batch_size]

                # Forward pass
                output = self.forward(batch_input)

                # Backward pass
                self.backward(output, batch_target, learning_rate)

            # Calculate and print the average loss for the epoch
            average_loss = np.mean([self.loss_function.loss(target[i], self.forward(input_data[i])) for i in range(num_samples)])
            print(f"Epoch {epoch}, Average Loss: {average_loss}")

    def predict_multi(self, input_data):
        probabilities = self.forward(input_data)[0]
        max_index = np.argmax(probabilities)
        output = np.zeros_like(probabilities)
        output[max_index] = 1
        output = output.astype(int)
        return output
    
    def predict_binary(self, input_data):
        probabilities = self.forward(input_data)[0]
        rounded = np.round(probabilities)
        output = rounded.astype(int)
        return output


class activationFunction:

    def activate(self,X):
        raise NotImplementedError("Abstract class.")

    def backprop_grad(self, grad):
        raise NotImplementedError("Abstract class.")

class Relu(activationFunction):
    def activate(self,X):
        return np.maximum(0, X)

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

    def backprop_grad(self, X):
        sig = self.activate(X)
        return  sig * (1 - sig)
    
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
    
class MeanSquaredError(LossFunction):
    def loss(self, Y, Yhat):
        return 0.5 * np.mean(np.square(Yhat - Y))

    def lossGradient(self, Y, Yhat):
        return Yhat - Y

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
    
class CrossEntropyLoss(LossFunction):
    def loss(self, Y, Yhat):
        # Avoid division by zero in log
        # epsilon = 1e-15
        # Yhat = np.clip(Yhat, epsilon, 1 - epsilon)
        # return -np.sum(Y * np.log(Yhat)) / len(Y)
        # return -np.sum(Y * np.log(Yhat + 10**-100))
        cce = tf.keras.losses.CategoricalCrossentropy()
        return cce(Y, Yhat).numpy()

    def lossGradient(self, Y, Yhat):
        # Avoid division by zero in gradient calculation
        # epsilon = 1e-15
        # Yhat = np.clip(Yhat, epsilon, 1 - epsilon)
        # return -(Y / Yhat) / len(Y)
        # return -Y/(Yhat + 10**-100)
        loss = self.loss(Y, Yhat)
        return tf.GradientTape().gradient(loss, Yhat)


def accuracy(yhat, y):
    ''' pred_label: (N,) vector; Y: (N,K) one hot encoded ground truth'''
    n = len(yhat)
    return sum(np.argmax(y[i]) == np.argmax(yhat[i]) for i in range(n)) * 1.0 / n


# nn = NeuralNetwork([2, 10, 1], [Sigmoid(), Sigmoid()], MeanSquaredError())

# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([[0], [1], [1], [0]])

# nn.train(X, y, epochs=100, learning_rate=.1)

# yhat = np.array([nn.forward(x) for x in X])
# print("Train accuracy: ", accuracy(yhat, y))