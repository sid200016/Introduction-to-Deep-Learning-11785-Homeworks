import numpy as np


class Upsample1d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        size_input = A.shape[2]
        size_output = self.upsampling_factor*(size_input-1)+1
        Z = np.zeros((A.shape[0], A.shape[1], size_output)) 

        # TODO
        Z[:, :, 0:size_output:self.upsampling_factor] = A
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = dLdZ[:, :, 0:dLdZ.shape[2]:self.upsampling_factor]
        return dLdA


class Downsample1d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        Z = A[:, :, 0:A.shape[2]:self.downsampling_factor]
        self.input_width = A.shape[2]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.input_width))
        dLdA[:, :, 0:self.input_width:self.downsampling_factor] = dLdZ
        # TODO
        return dLdA


class Upsample2d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        H_input = A.shape[2]
        H_output = self.upsampling_factor*(H_input-1)+1
        W_input = A.shape[3]
        W_output = self.upsampling_factor*(W_input-1)+1
        Z = np.zeros((A.shape[0], A.shape[1], H_output, W_output))
        
        # TODO
        Z[:, :, 0:H_output:self.upsampling_factor, 0:W_output:self.upsampling_factor] = A
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = dLdZ[:, :, 0:dLdZ.shape[2]:self.upsampling_factor, 0:dLdZ.shape[3]:self.upsampling_factor]
        return dLdA


class Downsample2d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        
        self.input_height = A.shape[2]
        self.input_width = A.shape[3]
        Z = A[:, :, 0:A.shape[2]:self.downsampling_factor, 0:A.shape[3]:self.downsampling_factor]# TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.input_height, self.input_width))
        dLdA[:, :, 0:self.input_height:self.downsampling_factor, 0:self.input_width:self.downsampling_factor] = dLdZ # TODO
        return dLdA