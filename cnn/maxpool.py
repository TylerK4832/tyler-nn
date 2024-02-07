import numpy as np

class MaxPool:

  '''
  Returns all 2x2 image regions for a given image
  '''
  def iterate_regions(self, image):
      h, w, _ = image.shape
      new_h, new_w = h // 2, w // 2

      for i in range(new_h):
          for j in range(new_w):
              im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
              yield im_region, i, j

  '''
  Completes a forward pass of a single image
  through the maxpool layer
  '''
  def forward(self, input):
      self.last_input = input

      h, w, num_filters = input.shape
      output = np.zeros((h // 2, w // 2, num_filters))

      for (im_region, i, j) in self.iterate_regions(input):
          output[i, j] = np.amax(im_region, axis=(0, 1))

      return output

  '''
  Completes a backward pass through the maxpool layer 
  using the loss gradient for the layer's outputs.
  '''
  def backprop(self, dLdout, lr):
      dLdinput = np.zeros(self.last_input.shape)

      for (im_region, i, j) in self.iterate_regions(self.last_input):
          h, w, f = im_region.shape
          amax = np.amax(im_region, axis=(0, 1))

          max_mask = (im_region == np.expand_dims(amax, axis=(0, 1)))

          dLdinput[i * 2:i * 2 + h, j * 2:j * 2 + w, :] += max_mask * np.expand_dims(dLdout[i, j, :], axis=(0, 1))

      return dLdinput
