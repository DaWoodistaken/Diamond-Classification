import numpy as np
from DL.bases import Layer, LayerWithWeights

class AffineLayer(LayerWithWeights):
    def __init__(self, input_size, output_size, seed=None):
        super(AffineLayer, self).__init__(input_size, output_size, seed=seed)

    def forward(self, x):
        '''
            :param x: activations/inputs from previous layer
            :return: output of affine layer
        '''
        out = None
        ##### YOUR CODE STARTS #####

        # Vectorize the input to [batchsize, others] array
        # Reshape x into 2D if needed for matrix multiplication
        x_reshaped = x.reshape(x.shape[0], -1)
        out = x_reshaped.dot(self.W) + self.b

        ##### YOUR CODE ENDS #######
    
        # Save x for using in backward pass
        self.x = x.copy()

        return out


    def backward(self, dprev):
        '''
            :param dprev: gradient of next layer:
            :return: downstream gradient
        '''

        batch_size = self.x.shape[0]
        x_vectorized = None
        dx, dw, db = None, None, None

        # YOUR CODE STARTS
        # Reshape x into 2D if needed for matrix multiplication
        x_vectorized = self.x.reshape(self.x.shape[0], -1)

        # Calculate gradients
        dw = x_vectorized.T.dot(dprev)
        db = np.sum(dprev, axis=0)
        dx = dprev.dot(self.W.T).reshape(self.x.shape)

        # YOUR CODE ENDS

        # Save them for backward pass
        self.db = db.copy()
        self.dW = dw.copy()
        return dx, dw, db

    def __repr__(self):
        return 'Affine layer'
