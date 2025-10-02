import numpy as np

class Flatten():
    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.input_shape = A.shape
        Z = np.reshape(A, (A.shape[0], A.shape[1]*A.shape[2]))  # TODO
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """
        dLdA = np.reshape(dLdZ, self.input_shape)  # TODO
        return dLdA

