import numpy as np

class Softmax:

  def __init__(self, input_len, nodes):
    self.weights = np.random.randn(input_len, nodes) / input_len
    self.biases = np.zeros(nodes)

  '''
  Computes the softmax probability of x
  '''
  def forward(self, x):
    self.last_x_shape = x.shape

    x = x.flatten()
    self.last_x = x

    input_len, nodes = self.weights.shape

    totals = np.dot(x, self.weights) + self.biases
    self.last_totals = totals

    totals -= np.max(totals)

    exp = np.exp(totals)
    return exp / np.sum(exp, axis=0)

  '''
  Completes a backwards pass through the softmax layer
  '''
  def backprop(self, dLdout, lr):
    for i, gradient in enumerate(dLdout):
      if gradient == 0:
        continue

      t_exp = np.exp(self.last_totals - np.max(self.last_totals))

      S = np.sum(t_exp)

      doutdt = -t_exp[i] * t_exp / (S ** 2)
      doutdt[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

      dtdw = self.last_x
      dtdb = 1
      dtdinputs = self.weights

      dLdt = gradient * doutdt

      d_L_d_w = dtdw[np.newaxis].T @ dLdt[np.newaxis]
      d_L_d_b = dLdt * dtdb
      d_L_d_inputs = dtdinputs @ dLdt

      self.weights -= lr * d_L_d_w
      self.biases -= lr * d_L_d_b

      return d_L_d_inputs.reshape(self.last_x_shape)

