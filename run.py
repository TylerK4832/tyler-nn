import mnist

import nn
from conv import Conv
from maxpool import MaxPool
from softmax import Softmax
from relu import ReLU
from sigmoid import Sigmoid
from dense import Dense

train_images = mnist.train_images()[:10000]
train_labels = mnist.train_labels()[:10000]
test_images = mnist.test_images()[:10000]
test_labels = mnist.test_labels()[:10000]

network = nn.nn(
  [
    Conv(32, 3),
    ReLU(),
    MaxPool(),

    Conv(64, 3),
    ReLU(),
    MaxPool(),

    Conv(64, 3),
    ReLU(),
    
    Dense(576, 64),
    ReLU(),

    Dense(64, 10),
    Softmax(10,10)
  ])

network.train(X=train_images, Y=train_labels, batch_size=200, epochs=5, initial_lr=0.001, decay_rate=0.8)
network.test(X=test_images, Y=test_labels)