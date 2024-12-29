import numpy as np

class Dense:
    def __init__(self, n_inputs, n_neurons, weight_l1_regularizer=0, weight_l2_regularizer=0, bias_l1_regularizer=0, bias_l2_regularizer=0):
        # Initialize the weights and biases
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros(shape=(1, n_neurons))

        # Initialize the regularization strength
        self.weight_l1_regularizer = weight_l1_regularizer
        self.weight_l2_regularizer = weight_l2_regularizer
        self.bias_l1_regularizer = bias_l1_regularizer
        self.bias_l2_regularizer = bias_l2_regularizer

    def forward(self, inputs):
        """
        Perform the forward pass through the layer.

        Parameters:
            inputs (ndarray): The input data to the layer. Shape (n_batches, n_inputs).
        """

        # Store the inputs for later use
        self.inputs = inputs
        # Calculate the output values from the inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dl_dz):
        """
        Perform the backward pass of the layer.

        Parameters:
            dl_dz (numpy.ndarray): P.Ds of the loss with respect to the output of this layer.
        """

        # gradients on parameters of this layer (weights and biases)
        self.dl_dw = np.dot(self.inputs.T, dl_dz)
        self.dl_db = np.sum(dl_dz, axis=0, keepdims=True)

        # gradient updates if regularization is used
        # L1 regularization on weights
        if self.weight_l1_regularizer > 0:
            self.dl_dw += self.weight_l1_regularizer * np.sign(self.weights)

        # L2 regularization on weights
        if self.weight_l2_regularizer > 0:
            self.dl_dw += 2 * self.weight_l2_regularizer * self.weights

        # L1 regularization on biases
        if self.bias_l1_regularizer > 0:
            self.dl_db += self.bias_l1_regularizer * np.sign(self.biases)
        
        # L2 regularization on biases
        if self.bias_l2_regularizer > 0:
            self.dl_db += 2 * self.bias_l2_regularizer * self.biases

        # gradients on the inputs for this layer
        self.dl_dX = np.dot(dl_dz, self.weights.T)

class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        
    def forward(self, inputs):
        """
        Perform the forward pass of the dropout layer.

        Parameters:
            inputs (numpy.ndarray): Input data to the layer. Shape (n_batches, n_inputs).
        """
        # save the inputs
        self.inputs = inputs 
        # save the binary mask
        self.binary_mask = np.random.binomial(n=1, p=1-self.dropout_rate, size=inputs.shape) / (1-self.dropout_rate)
        # apply the mask to the inputs
        self.output = inputs * self.binary_mask

    def backward(self, dl_dz):
        """
        Perform the backward pass of the layer.

        Parameters:
            dl_dz (numpy.ndarray): Gradient of the loss with respect to the output of this layer.
        """
        # calculate the gradient on the inputs
        self.dl_dX = dl_dz * self.binary_mask 