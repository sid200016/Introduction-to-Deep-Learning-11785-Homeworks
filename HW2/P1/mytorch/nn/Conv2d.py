import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        Z = np.zeros((len(A), self.out_channels, A.shape[2] - self.kernel_size + 1, A.shape[3] - self.kernel_size + 1))
        for i in range(len(A)):
            for out_c in range(self.out_channels): 
                for j in range(Z.shape[2]): 
                    for k in range(Z.shape[3]):
                        Z[i, out_c, j, k] = np.sum(A[i, :, j:j+self.kernel_size, k:k+self.kernel_size] *self.W[out_c, :, :, :]) + self.b[out_c]
        self.Z = Z
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        self.dLdW = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))  # TODO
        self.dLdb = np.zeros((self.out_channels, ))  # TODO
        dLdA = np.zeros((self.A.shape[0], self.in_channels, self.A.shape[2], self.A.shape[3]))  # TODO
        #dLdB
        for batch in range(dLdZ.shape[0]):
            for oc in range(self.out_channels):
                self.dLdb[oc] += np.sum(dLdZ[batch, oc, :, :])
        #dLdW
        for batch in range(dLdZ.shape[0]):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    for i in range(self.kernel_size):
                        for j in range(self.kernel_size):
                            self.dLdW[oc, ic, i, j] += np.sum(self.A[batch, ic, i:i+dLdZ.shape[2], j:j+dLdZ.shape[3]] * dLdZ[batch, oc, :, :])
        #dLdA
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)), mode='constant')
        weight_flipped = np.flip(self.W, axis = (2, 3))
        for i in range(len(self.A)):
            for in_c in range(self.in_channels):
                for j in range(self.A.shape[2]):
                    for k in range(self.A.shape[3]):
                        window = dLdZ_padded[i, :, j:j+self.kernel_size, k: k+self.kernel_size]
                        dLdA[i, in_c, j, k] += np.sum(window *weight_flipped[:, in_c, :, :])
        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Pad the input appropriately using np.pad() function
        # TODO
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')
        # Call Conv2d_stride1
        # TODO
        Z = self.conv2d_stride1.forward(A_padded)
        # downsample
        Z_down = self.downsample2d.forward(Z)  # TODO

        return Z_down

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Call downsample1d backward
        # TODO
        dLdZ_upsampled = self.downsample2d.backward(dLdZ)  # TODO
        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ_upsampled)  # TODO

        # Unpad the gradient
        # TODO
        dLdA_unpadded = dLdA[:, :, self.pad:dLdA.shape[2]-self.pad, self.pad:dLdA.shape[3] - self.pad]
        return dLdA_unpadded
