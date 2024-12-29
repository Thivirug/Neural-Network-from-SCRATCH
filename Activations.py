import numpy as np

class ReLU:
    def forward(self, inputs):
        """
        Perform the forward pass of the activation function.

        Parameters:
            inputs (numpy.ndarray): Input data to the activation function. Shape (n_batches, n_neurons(of the previous layer)).
        """

        # Store the inputs for later use
        self.inputs = inputs
        # Calculate the output values from the inputs
        self.output = np.maximum(0, inputs, dtype=float) 

    def backward(self, dl_dz):
        """
        Performs the backward pass of the activation function.

        Parameters:
            dl_dz (numpy.ndarray): P.Ds of the loss with respect to the ReLU outputs.
        """

        # copy of dl_dz since we modify it
        self.dl_dInputs = dl_dz.copy()
        # where inputs values are negative, set position in dl_dz to 0
        self.dl_dInputs[self.inputs <= 0] = 0

class Softmax:
    def forward(self, inputs):
        """
        Perform the forward pass for the SoftMax activation function.

        Parameters:
            inputs (ndarray): Input data, typically the output of the previous layer.
        """
        # Subtract the maximum value from the inputs to prevent overflow in the exponential function (unnormalized log probabilities)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize the values
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities