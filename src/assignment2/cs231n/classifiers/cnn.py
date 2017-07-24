from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        C, H, W = input_dim
        stride = conv_param['stride']
        pad = conv_param['pad']
        H_prime = int(1 + (H + 2 * pad - filter_size) / stride)
        W_prime = int(1 + (W + 2 * pad - filter_size) / stride)
        HH = pool_param['pool_height']
        WW = pool_param['pool_width']
        stride = pool_param['stride']
        H_prime = int(1 + (H_prime - HH) / stride)
        W_prime = int(1 + (W_prime - WW) / stride)
        W2_dim = H_prime*H_prime*num_filters

        
        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(num_classes)
        self.params['W1'] = weight_scale*np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['W2'] = weight_scale*np.random.randn(W2_dim,hidden_dim)
        self.params['W3'] = weight_scale*np.random.randn(hidden_dim,num_classes)

        print(self.params['b1'].shape)
        print(self.params['b2'].shape)
        print(self.params['b3'].shape)
        print(self.params['W1'].shape)
        print(self.params['W2'].shape)
        print(self.params['W3'].shape)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.params['W1'].shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        # N, C, H, W = X.shape
        # stride = conv_param['stride']
        # pad = conv_param['pad']
        # H_prime = int(1 + (H + 2 * pad - filter_size) / stride)
        # W_prime = int(1 + (W + 2 * pad - filter_size) / stride)
        # HH = pool_param['pool_height']
        # WW = pool_param['pool_width']
        # stride = pool_param['stride']
        # H_prime = int(1 + (H_prime - HH) / stride)
        # W_prime = int(1 + (W_prime - WW) / stride)
        # W2_dim = H_prime*H_prime*self.params['W1'].shape[0]
        # self.params['W2'] = weight_scale*np.random.randn(hidden_dim,hidden_dim)

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #conv - relu - 2x2 max pool - affine - relu - affine - softmax

        ############################################################################
        out_1, cache_1 = conv_relu_pool_forward(X, W1, b1,conv_param, pool_param)
        out_2, cache_2 = affine_relu_forward(out_1, W2, b2)
        out_3, cache_3 = affine_forward(out_2, W3, b3)
        scores = out_3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        ############################################################################
        loss, dx = softmax_loss(out_3, y)
        loss += 0.5 * self.reg * (np.sum(self.params['W1']*self.params['W1']) + np.sum(self.params['W2']*self.params['W2'])+ np.sum(self.params['W3']*self.params['W3']))
        dx, grads['W3'], grads['b3'] = affine_backward(dx, cache_3)
        dx, grads['W2'], grads['b2'] = affine_relu_backward(dx, cache_2)
        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, cache_1)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
