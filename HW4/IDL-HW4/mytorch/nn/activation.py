import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        Z_shifted = Z - np.max(Z, axis = self.dim, keepdims=True)
        self.Z_s = Z_shifted
        self.A = np.exp(Z_shifted)/np.sum(np.exp(Z_shifted), axis=self.dim, keepdims=True) # TODO
        return self.A

        


    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
        A = np.moveaxis(self.A, self.dim, -1)
        dLdA = np.moveaxis(dLdA, self.dim, -1)

        A_flat = A.reshape(-1, A.shape[-1])
        dLdA_flat = dLdA.reshape(-1, dLdA.shape[-1])

        dot = np.sum(dLdA_flat * A_flat, axis=1, keepdims=True)
        dLdZ_flat = A_flat * (dLdA_flat - dot)

    # Reshape back to the shape after moveaxis (before flattening)
        dLdZ = dLdZ_flat.reshape(A.shape)

    # Move softmax axis back to original position
        dLdZ = np.moveaxis(dLdZ, -1, self.dim)
        return dLdZ

    