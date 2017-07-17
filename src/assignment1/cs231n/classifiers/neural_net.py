from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

def f_hinge(X):
  X[X<0] = 0
  return X

def df_hinge(X, dF):
  X[X<0] = 0
  X[X>0] = 1
  return dF*X

def f_softmax(X, y):
  X -= np.amax(X,axis=1)[:,np.newaxis]
  return -X[np.arange(X.shape[0]),y] + np.log(np.sum(np.exp(X), axis=1))

def df_softmax(X, y, dF):
  dX = np.exp(X)/np.sum(np.exp(X),axis=1)[:,np.newaxis]
  dX[np.arange(X.shape[0]),y] += -1
  return dX*dF

def df_matrix_multiplication(A,B,dF):
  dA = dF.dot(B.T) #.T gives the transpose of the matrix
  dB = A.T.dot(dF)
  return dA, dB

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    # scores = hinge_f(X.dot(W1)+b1).dot(W2) + b2
    A1 = X.dot(W1)
    Ab1 = A1 + b1
    Z1 = f_hinge(Ab1)
    A2 = Z1.dot(W2)
    Ab2 = A2 + b2

    scores = Ab2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    # S = scores.copy()
    # S -= np.amax(S,axis=1)[:,np.newaxis]
    # Z2 = -S[np.arange(N),y]+np.log(np.sum(np.exp(S),axis=1))
    # loss = np.sum(Z2)
    # loss /= N
    # loss += reg * (np.sum(W1*W1) + np.sum(W2*W2))
    Z2 = f_softmax(Ab2,y)
    loss = np.sum(Z2)/N + reg * (np.sum(W1*W1) + np.sum(W2*W2))

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}

    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################

    d_loss = 1
    d_Z2 = d_loss/N
    d_Ab2 = df_softmax(Ab2,y,d_Z2)
    grads['b2'] = np.sum(d_Ab2,axis=0)
    d_A2 = d_Ab2
    d_Z1, grads['W2'] = df_matrix_multiplication(Z1, W2, d_A2)
    d_Ab1 = df_hinge(Ab1, d_Z1)
    grads['b1'] = np.sum(d_Ab1,axis=0)
    d_A1 = d_Ab1
    d_X, grads['W1'] = df_matrix_multiplication(X, W1, d_A1)
    grads['W1'] += 2*reg*W1
    grads['W2'] += 2*reg*W2
    # dZ2dAb2 = (1-Z2)*Z2
    # dAb2db2 = np.eye(dZ2dAb2.T.shape[0])
    # dA2dW2 = Z1
    # dAb2dA2 = np.eye(dA2dW2.T.shape[0],dA2dW2.T.shape[1])
    # dA2dZ1 = W2
    # Ab1C = Ab1.copy()
    # Ab1C[Ab1C>0] = 1
    # Ab1C[Ab1C<0] = 0
    # dZ1dAb1 = Ab1C
    # dAb1db1 = np.eye(dZ1dAb1.T.shape[0],dZ1dAb1.T.shape[1])
    # dAb1dA1 = np.eye(dZ1dAb1.T.shape[0],dZ1dAb1.T.shape[1])
    # dA1dW1 = X
    # dA1dX = W1

    # grads['b2'] = dAb2db2.dot(dZ2dAb2)
    # grads['W2'] = dA2dW2.dot(dAb2dA2).dot(dZ2dAb2)
    # grads['b1'] = dAb1db1.dot(dZ1dAb1).dot(dA2dZ1).dot(dAb2dA2).dot(dZ2dAb2)
    # grads['W1'] = dA1dW1.dot(dAb1dA1).dot(dZ1dAb1).dot(dA2dZ1).dot(dAb2dA2).dot(dZ2dAb2)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      sample_list = np.random.choice(num_train,batch_size)
      X_batch = X[sample_list]
      y_batch = y[sample_list]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      drop_out = -0.001
      W1_mask = np.random.randn(self.params['W1'].shape[0],self.params['W1'].shape[1])
      W2_mask = np.random.randn(self.params['W2'].shape[0],self.params['W2'].shape[1])
      W1_mask[W1_mask<=drop_out] = 0
      W1_mask[W1_mask>drop_out] = 1
      W2_mask[W2_mask<=drop_out] = 0
      W2_mask[W2_mask>drop_out] = 1

      W1_mask = np.ones((self.params['W1'].shape[0],self.params['W1'].shape[1]))
      W2_mask = np.ones((self.params['W2'].shape[0],self.params['W2'].shape[1]))

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] -= W1_mask*learning_rate*grads['W1']
      self.params['W2'] -= W2_mask*learning_rate*grads['W2']
      self.params['b1'] -= learning_rate*grads['b1']
      self.params['b2'] -= learning_rate*grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy  
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    y_pred = np.argmax(f_hinge(X.dot(self.params['W1'])+self.params['b1']).dot(self.params['W2'])+self.params['b2'],axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred
