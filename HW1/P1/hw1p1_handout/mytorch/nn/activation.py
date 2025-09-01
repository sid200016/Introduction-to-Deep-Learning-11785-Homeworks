import numpy as np
import scipy
from scipy.special import erf

### No need to modify Identity class
class Identity:
    """
    Identity activation function.
    """

    def forward(self, Z):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.A = Z
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ
        return dLdZ


class Sigmoid:
    """
    Sigmoid activation function.

    TODO:
    On same lines as above, create your own mytorch.nn.Sigmoid!
    Define 'forward' function.
    Define 'backward' function.
    Read the writeup (Hint: Sigmoid Section) for further details on Sigmoid forward and backward expressions.
    """
    def forward(self, Z):
        self.A  = 1/(1+np.exp(-Z))
        return self.A
    def backward(self, dLdA):
        dAdz = dLdA * (self.A - self.A * self.A)
        dLdZ = dLdA * dAdz
        return dLdZ

class Tanh:
    """
    Tanh activation function.

    TODO:
    On same lines as above, create your own mytorch.nn.Tanh!
    Define 'forward' function.
    Define 'backward' function.
    Read the writeup (Hint: Tanh Section) for further details on Tanh forward and backward expressions.
    """
    def forward(self, Z):
        self.A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        return self.A
    def backward(self, dLdA):
        dAdZ = 1 - np.square(self.A)
        dLdZ = dLdA * dAdZ
        return dLdZ 
class ReLU:
    """
    ReLU (Rectified Linear Unit) activation function.

    TODO:
    On same lines as above, create your own mytorch.nn.ReLU!
    Define 'forward' function.
    Define 'backward' function.
    Read the writeup (Hint: ReLU Section) for further details on ReLU forward and backward expressions.
    """
    def forward(self, Z):
        # self.Z = Z
        self.A = np.maximum(0, Z)
        return self.A
    def backward(self, dLdA):
        dAdZ = np.where(self.A > 0, 1.0, 0.0)
        dLdZ = dLdA * dAdZ
        return dLdZ

class GELU:
    """
    GELU (Gaussian Error Linear Unit) activation function.

    TODO:
    On same lines as above, create your own mytorch.nn.GELU!
    Define 'forward' function.
    Define 'backward' function.
    Read the writeup (Hint: GELU Section) for further details on GELU forward and backward expressions.
    Note: Feel free to save any variables from gelu.forward that you might need for gelu.backward.
    """
    def forward(self, Z):
        self.Z = Z
        self.A = 0.5*Z*(1 + erf(Z/np.sqrt(2)))
        return self.A
    
    def backward(self, dLdA):
        dAdZ = 0.5*(1+erf(self.Z/np.sqrt(2))) +  self.Z*np.exp(-np.square(self.Z)/2) / np.sqrt(2*np.pi)
        dLdZ = dLdA * dAdZ
        return dLdZ

class Swish:
    """
    Swish activation function.

    TODO:
    On same lines as above, create your own Swish which is a torch.nn.SiLU with a learnable parameter (beta)!
    Define 'forward' function.
    Define 'backward' function.
    Read the writeup (Hint: Swish Section) for further details on Swish forward and backward expressions.
    """
    def __init__(self, B = 1.0):
        self.beta = B
    def forward(self, Z):
        self.Z = Z
        self.A = Z * 1/(1+np.exp(-Z*self.beta))
    def backward(self, dLdA):
        dAdZ = 1/(1+np.exp(-self.Z*self.beta)) + self.beta*self.Z*(1+np.exp(-self.Z*self.beta))*(1 - 1/(1+np.exp(-self.Z*self.beta)))
        dLdZ = dLdA * dAdZ
        return dLdZ

class Softmax:
    """
    Softmax activation function.

    ToDO:
    On same lines as above, create your own mytorch.nn.Softmax!
    Complete the 'forward' function.
    Complete the 'backward' function.
    Read the writeup (Hint: Softmax Section) for further details on Softmax forward and backward expressions.
    Hint: You read more about `axis` and `keep_dims` attributes, helpful for future homeworks too.
    """

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        Note: How can we handle large overflow values? Hint: Check numerical stability.
        """
        
        Z_shifted = Z - np.max(Z, axis = 1, keepdims=True)
        self.Z_s = Z_shifted
        self.A = np.exp(Z_shifted)/np.sum(np.exp(Z_shifted), axis=1, keepdims=True) # TODO
        return self.A

    def backward(self, dLdA):
        # Calculate the batch size and number of features
        N = self.Z_s.shape[0] # TODO
        C = self.Z_s.shape[1]  # TODO

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros_like(dLdA)  # TODO

        # Fill dLdZ one data point (row) at a time.
        for i in range(N):
            # Initialize the Jacobian with all zeros.
            # Hint: Jacobian matrix for softmax is a _×_ matrix, but what is _ here?
            J = np.zeros((C, C))  # TODO

            # Fill the Jacobian matrix, please read the writeup for the conditions.
            a = self.A[i, :]
            for m in range(C):
                for n in range(C):
                    if m == n:
                        J[m, n] = a[m]*(1 - a[n])
                    else:
                        J[m, n] = a[m]*a[n]  # TODO


            # Calculate the derivative of the loss with respect to the i-th input, please read the writeup for it.
            # Hint: How can we use (1×C) and (C×C) to get (1×C) and stack up vertically to give (N×C) derivative matrix?
            dLdZ[i, :] = np.dot(dLdA[i, :], J)  # TODO

        return dLdZ # TODO - What should be the return value?
