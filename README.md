# A Basic Neural Network Framework

A framework to easily create and customize deep neural networks including 
convolutional neural networks (CNNs) for computer vision applications.

Classes are provided for common layer types including dense, convolutional,
and maxpool layers. There are also classes provided for a number of activation
functions.

## Use

As an example, we will build a neural network to classify the MNIST dataset.
First, import the necessary libraries.

```python
import mnist
import nn
from conv import Conv
from maxpool import MaxPool
from softmax import Softmax
from relu import ReLU
from sigmoid import Sigmoid
from dense import Dense
```
Then, initialize your train and test data.

```python
train_images = mnist.train_images()[:10000]
train_labels = mnist.train_labels()[:10000]
test_images = mnist.test_images()[:2000]
test_labels = mnist.test_labels()[:2000]
```

Define a neural network architecture by building a sequence of layers:

```python
network = nn.nn(
  [
    Conv(num_filters=32, filter_size=3),
    ReLU(),
    MaxPool(),

    Conv(num_filters=64, filter_size=3),
    ReLU(),
    MaxPool(),

    Conv(num_filters=64, filter_size=3),
    ReLU(),
    
    Dense(input_size=576, output_size=64),
    ReLU(),

    Dense(input_size=64, output_size=10),
    Softmax(input_size=10,output_size=10)
  ])
```
Finally, train and test your network with the provided methods:

```python
network.train(X=train_images, Y=train_labels, batch_size=200, epochs=5, initial_lr=0.001, decay_rate=0.8)
network.test(X=test_images, Y=test_labels)
```
