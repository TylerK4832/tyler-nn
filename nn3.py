from dataclasses import dataclass # for neural network construction
import pickle                     # to import MNIST data
import gzip                       # to import MNIST data
import random                     # to initialize weights and biases
import numpy as np                # for all needed math
from PIL import Image, ImageOps   # for image file processing
from time import time             # for performance measurement

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def cost_derivative(output_activations, y):
    return (output_activations - y) 

@dataclass
class Network:
    num_layers: int
    biases: list
    weights: list

def init_network(layers):
    
    return Network(
        len(layers),
        
        # input layer doesn't have biases
        [np.random.randn(y, 1) for y in layers[1:]],
        
        # there are no (weighted) connections into input layer or out of the output layer
        [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
    )

def feedforward(nn, a):
    for b, w in zip(nn.biases, nn.weights):
        a = sigmoid(np.dot(w, a) + b)
    return a

def evaluate(nn, test_data):
    test_results = [(np.argmax(feedforward(nn, x)), y) for (x, y) in test_data]
    
    return sum(int(x == y) for (x, y) in test_results)

def learn(nn, training_data, epochs, mini_batch_size, learning_rate, test_data = None):
    n = len(training_data)

    for j in range(epochs):
        random.shuffle(training_data) # that's where "stochastic" comes from

        mini_batches = [
            training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)
        ]
        
        for mini_batch in mini_batches:
            stochastic_gradient_descent(nn, mini_batch, learning_rate) # that's where learning really happes

        if test_data:
            print('Epoch {0}: accuracy {1}%'.format(f'{j + 1:2}', 100.0 * evaluate(nn, test_data) / len(test_data)))
        else:
            print('Epoch {0} complete.'.format(f'{j + 1:2}'))


def stochastic_gradient_descent(nn, mini_batch, eta):
    # "nabla" is the gradient symbol
    nabla_b = [np.zeros(b.shape) for b in nn.biases]
    nabla_w = [np.zeros(w.shape) for w in nn.weights]
    
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = backprop(nn, x, y) # compute the gradient
        
        # note that here we call the return values 'delta_nabla', while in the
        # backprop function we call them 'nabla'

        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        
    nn.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(nn.weights, nabla_w)]
    nn.biases  = [b - (eta / len(mini_batch)) * nb for b, nb in zip(nn.biases, nabla_b)]


def backprop(nn, x, y):
    nabla_b = [np.zeros(b.shape) for b in nn.biases]
    nabla_w = [np.zeros(w.shape) for w in nn.weights]

    # feedforward
    activation = x    # first layer activation is just its input
    activations = [x] # list to store all activations, layer by layer
    zs = []           # list to store all z vectors, layer by layer

    for b, w in zip(nn.biases, nn.weights):
        z = np.dot(w, activation) + b  # calculate z for the current layer
        zs.append(z)                   # store
        activation = sigmoid(z)        # layer output
        activations.append(activation) # store

    # backward pass

    # 1. starting from the output layer
    delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) 
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    # 2. continue back to the input layer (i is the layer index, we're using i instead of l
    #    to improve readability -- l looks too much like 1)
    for i in range(2, nn.num_layers): # starting from the next-to-last layer
        z = zs[-i]
        sp = sigmoid_prime(z)
        delta = np.dot(nn.weights[-i + 1].transpose(), delta) * sp
        
        nabla_b[-i] = delta
        nabla_w[-i] = np.dot(delta, activations[-i - 1].transpose())
        
    return (nabla_b, nabla_w)


def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="bytes")
    f.close()
    
    return (training_data, validation_data, test_data)