import numpy as np

class Conv:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size

        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size ** 2)

    '''
    Returns all image regions for a given image
    '''
    def iterate_regions(self, x):
        h, w = x.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                yield x[i:(i + self.filter_size), j:(j + self.filter_size)], i, j

    '''
    Completes a forward pass of a single image
    through the convolutional layer and returns 
    the output
    '''
    def forward(self, x):
        self.last_x = x
        h, w, _ = x.shape

        output = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))

        for (x_region, i, j) in self.iterate_regions(x[:, :, 0]):
            output[i, j, :] = np.sum(x_region * self.filters, axis=(1, 2))

        return output

    '''
    Completes a backward pass through the convolutional
    layer and updates filters accordingly using the 
    loss gradient for the layer's outputs.
    '''
    def backprop(self, dLdout, lr):
        dLdfilters = np.zeros(self.filters.shape)
        dLdinput = np.zeros(self.last_x.shape)

        for (im_region, i, j) in self.iterate_regions(self.last_x[:, :, 0]):
            dLdfilters[:, :, :] += np.expand_dims(dLdout[i, j, :], axis=(1, 2)) * im_region
            dLdinput[i:(i + self.filter_size), j:(j + self.filter_size), 0] += np.sum(self.filters * np.expand_dims(dLdout[i, j, :], axis=(1, 2)), axis=0)

        self.filters -= lr * dLdfilters

        return dLdinput