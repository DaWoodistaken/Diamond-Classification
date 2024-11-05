import numpy as np
from DL.bases import Layer


class ExponentialLinearUnit(Layer):
    def __init__(self, alpha=1e-3):
        self.x = None
        self.alpha = alpha

    def forward(self, x):
        '''
            Forward pass for ELU
            :param x: outputs of previous layer
            :return: ELU activation
        '''
        self.x = x.copy()
        activated_x = np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))
        return activated_x

    def backward(self, dprev):
        '''
            Backward pass of ELU
            :param dprev: gradient of previous layer
            :return: upstream gradient
        '''
        # Check if dprev is a tuple and extract the first element if it is
        if isinstance(dprev, tuple):
            dprev = dprev[0]

        # Ensure dprev is an ndarray
        dprev = np.array(dprev)

        # Debugging statements to inspect shapes
        print(f"Shape of dprev: {dprev.shape}")
        print(f"Shape of self.x: {self.x.shape}")

        # Ensure shapes are consistent
        if dprev.shape != self.x.shape:
            dprev = dprev.reshape(self.x.shape)

        # Compute gradient
        dx = np.where(self.x >= 0, dprev, self.alpha * np.exp(self.x) * dprev)
        return dx




class ReLU(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dprev):
        # Check if dprev is a tuple and extract the first element if it is
        if isinstance(dprev, tuple):
            dprev = dprev[0]

        # Ensure shapes of `self.x` and `dprev` are consistent
        if dprev.shape != self.x.shape:
            dprev = dprev.reshape(self.x.shape)

        # Apply the ReLU gradient: pass gradients only where x > 0
        dx = np.where(self.x > 0, dprev, 0)
        return dx

class Softplus(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.log(1 + np.exp(x))

    def backward(self, dprev):
        # Check if dprev is a tuple and extract the first element if it is
        if isinstance(dprev, tuple):
            dprev = dprev[0]

        # Ensure `dprev` has the same shape as `self.x`
        if dprev.shape != self.x.shape:
            dprev = dprev.reshape(self.x.shape)

        # Compute the gradient of the Softplus function
        sigmoid_x = 1 / (1 + np.exp(-self.x))
        dx = sigmoid_x * dprev
        return dx

class Swish(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return x / (1 + np.exp(-x))

    def backward(self, dprev):
        # Check if dprev is a tuple and extract the first element if it is
        if isinstance(dprev, tuple):
            dprev = dprev[0]

        # Ensure `dprev` has the same shape as `self.x`
        if dprev.shape != self.x.shape:
            dprev = dprev.reshape(self.x.shape)

        # Compute the gradient of the Swish function
        sigmoid_x = 1 / (1 + np.exp(-self.x))
        dx = (sigmoid_x + self.x * sigmoid_x * (1 - sigmoid_x)) * dprev
        return dx