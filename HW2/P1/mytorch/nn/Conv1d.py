# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        Z = np.zeros((len(A), self.out_channels, A.shape[2] - self.kernel_size + 1))
        print(Z.shape)
        for i in range(len(A)):
            for out_c in range(self.out_channels): 
                for j in range(Z.shape[2]): 
                    Z[i, out_c, j] = np.sum(A[i, :, j:j+self.kernel_size] *self.W[out_c, :, :]) + self.b[out_c]
            
        

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # TODO
        self.dLdb = np.zeros(dLdZ.shape[1])# TODO
        for batch in range(dLdZ.shape[0]):
            for oc in range(self.out_channels):
                self.dLdb[oc] += np.sum(dLdZ[batch, oc, :])
        
        self.dLdW = np.zeros((self.out_channels, self.in_channels, self.kernel_size))# TODO
        for i in range(len(self.A)):
            for out_c in range(self.out_channels): 
                for in_c in range(self.in_channels):
                    for k in range(self.kernel_size):
                        self.dLdW[out_c, in_c, k] += np.sum(self.A[i, in_c, k:k+dLdZ.shape[2]] * dLdZ[i, out_c, :])
        
        dLdZ_expanded = np.expand_dims(dLdZ, axis=1)  
        dLdZ_broadcasted = np.repeat(dLdZ_expanded, self.in_channels, axis=1)  
        dLdA = np.zeros((dLdZ.shape[0], self.in_channels, self.A.shape[2])) # TODO
        dLdZ_padded = np.pad(dLdZ_broadcasted, ((0,0),(0,0),(0,0),(self.kernel_size-1,self.kernel_size-1)), mode='constant') 
        W_flipped = np.flip(self.W, axis=2)
        for i in range(len(self.A)):
            for in_c in range(self.in_channels):
                for j in range(self.A.shape[2]):
                
                    window = dLdZ_padded[i, in_c, :, j:j+self.kernel_size]
                    dLdA[i, in_c, j] = np.sum(window * W_flipped[:, in_c, :])
  


        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0, weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding
        
        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample1d = Downsample1d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)  
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # Pad the input appropriately using np.pad() function
        # TODO
        A_padded = np.pad(A, ((0,0),(0,0),(self.pad,self.pad)), mode='constant')
        Z = self.conv1d_stride1.forward(A_padded)
        # Call Conv1d_stride1
        # TODO

        # downsample
        Z_down = self.downsample1d.forward(Z)  # TODO
        
        return Z_down
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ_upsampled = self.downsample1d.backward(dLdZ)  # TODO

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ_upsampled)  # TODO

        # Unpad the gradient
        # TODO
        dLdA_unpadded = dLdA[:, :, self.pad : dLdA.shape[2] - self.pad]

        return dLdA_unpadded
