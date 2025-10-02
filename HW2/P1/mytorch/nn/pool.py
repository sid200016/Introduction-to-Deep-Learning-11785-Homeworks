import numpy as np
from resampling import *


class MaxPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        self.A = A
        self.indices = np.zeros((A.shape[0], A.shape[1], A.shape[2] - self.kernel + 1, A.shape[3] - self.kernel + 1, 2), dtype=int)
        self.prev_max_val = -np.inf
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = np.zeros((A.shape[0], A.shape[1], A.shape[2] - self.kernel + 1, A.shape[3] - self.kernel + 1))
        for batch in range(len(A)):
            for channel in range(A.shape[1]):
                for i in range(A.shape[2] - self.kernel + 1):
                    for j in range(A.shape[3] - self.kernel + 1):
                        window = A[batch, channel, i:i+self.kernel, j:j+self.kernel]
                        max_val = np.argmax(window)
                        imax, jmax = np.unravel_index(max_val, window.shape)
                        Z[batch, channel, i, j] = window[imax, jmax]
                        self.indices[batch, channel, i, j, 0] = i + imax
                        self.indices[batch, channel, i, j, 1] = j + jmax
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros_like(self.A)
        for batch in range(dLdZ.shape[0]):
            for channel in range(dLdZ.shape[1]):
                for i in range(dLdZ.shape[2]):
                    for j in range(dLdZ.shape[3]):
                        imax = self.indices[batch, channel, i, j, 0]
                        jmax = self.indices[batch, channel, i, j, 1]
                        dLdA[batch, channel, imax, jmax] += dLdZ[batch, channel, i, j]
        return dLdA


class MeanPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        Z = np.zeros((A.shape[0], A.shape[1], A.shape[2] - self.kernel + 1, A.shape[3] - self.kernel + 1))
        for batch in range(len(A)):
            for channel in range(A.shape[1]):
                for i in range(A.shape[2] - self.kernel + 1):
                    for j in range(A.shape[3] - self.kernel + 1):
                        window  = A[batch, channel, i:i+self.kernel, j:j+self.kernel]
                        Z[batch, channel, i, j] = np.mean(window)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros_like(self.A)
        for batch in range(dLdZ.shape[0]):
            for channel in range(dLdZ.shape[1]):
                for i in range(dLdZ.shape[2]):
                    for j in range(dLdZ.shape[3]):
                        dLdA[batch, channel, i:i+self.kernel, j:j+self.kernel] += dLdZ[batch, channel, i, j] / (self.kernel *self.kernel)
        return dLdA


class MaxPool2d():
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z_up = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z_up)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)

        """
        dLdA_down = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdA_down)
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z_up = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z_up)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA_down = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdA_down)
        return dLdA
