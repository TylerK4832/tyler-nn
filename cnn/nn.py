import numpy as np
import time

class nn:

  def __init__(self, layers):
    self.layers = layers
    
  '''
  Completes a forward pass of a single image
  through the network's layers and returns 
  the categorical cross entropy loss and
  accuracy
  '''
  def forward(self, x, y):
    x = np.expand_dims(x, axis=-1)

    out = self.layers[0].forward((x / 255) - 0.5)
    for layer in self.layers[1:]:
      out = layer.forward(out)

    epsilon = 1e-15
    loss = -np.log(out[y] + epsilon)
    acc = 1 if np.argmax(out) == y else 0

    return out, loss, acc

  '''
  Executes a training step for a single 
  image and label. First obtain the loss from
  forward propogation, then backpropogate through
  the layers and update the weights
  '''
  def train_step(self, x, y, lr):
    
    out, loss, acc = self.forward(x, y)

    epsilon = 1e-15
    gradient = np.zeros(10)
    gradient[y] = -1 / (out[y] + epsilon)

    for layer in self.layers[::-1]:
      gradient = layer.backprop(gradient, lr)

    return loss, acc
  
  '''
  Train a single batch and return the average loss
  and accuracy
  '''
  def train_batch(self, X, Y, lr):
    batch_size = len(X)
    total_loss = 0
    total_acc = 0

    for i in range(batch_size):
      loss, acc = self.train_step(X[i], Y[i], lr)
      total_loss += loss
      total_acc += acc

    avg_loss = total_loss / batch_size
    avg_acc = total_acc / batch_size

    return avg_loss, avg_acc
  
  '''
  Train all batches, epochs and output loss and accuracy
  '''
  def train(self, X, Y, batch_size, epochs, initial_lr=0.001, decay_rate=0.9):
    total_start_time = time.time()
    for epoch in range(epochs):
      print('--- Epoch %d ---' % (epoch + 1))

      avg_loss, avg_acc = 0, 0

      batch_num = 0
      for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        Y_batch = Y[i:i+batch_size]
        batch_start_time = time.time()
        l, acc = self.train_batch(X_batch, Y_batch, initial_lr)
        batch_time = time.time() - batch_start_time
        avg_loss += l
        avg_acc += acc
        batch_num += 1

        print(
          '[Batch %d]: Average Loss %.3f | Accuracy: %.2f%% | Time: %.2f seconds' %
          (batch_num,l, acc * 100, batch_time)
        )

      avg_loss /= len(X) / batch_size
      avg_acc /= len(X) / batch_size
      
      initial_lr *= decay_rate

      print('Average Training Loss: %.3f | Accuracy: %.2f%%' % (avg_loss, avg_acc * 100))

    total_time = time.time() - total_start_time
    print("\nTotal training time: %.2f seconds" % total_time)


  '''
  Forward propogate images through the network,
  calculating average loss and accuracy
  '''
  def test(self, X, Y):
    print('\n--- Testing the CNN ---')

    loss = 0
    num_correct = 0
    for im, label in zip(X, Y):
      _, l, acc = self.forward(im, label)
      loss += l
      num_correct += acc

    num_tests = len(X)
    print('Test Loss:', loss / num_tests)
    print('Test Accuracy:', num_correct / num_tests)