# Implement the Softmax and CrossEntropy loss functions combined.
from Activations import Softmax
from Losses import CategoricalCrossEntropy, SparseCategoricalCrossEntropy
import numpy as np

class Softmax_Sparse:
    def __init__(self):
        self.softmax = Softmax()
        self.sparse_categorical_crossentropy = SparseCategoricalCrossEntropy()

    def forward(self, inputs, y_true, dense_layers=None):
        """
        Perform the forward pass through the softmax activation function and calculate the average loss.

        Parameters:
            inputs (ndarray): The input data to the softmax activation function.
            y_true (ndarray): The true labels for the input data (Integer values).
            dense_layers (list): List of Dense layer objects to. Default is None.

        Returns:
            float: The average loss value for the batch.
        """

        # Perform the forward pass through the softmax activation function
        self.softmax.forward(inputs)
        self.y_preds = self.softmax.output

        # Calculate the loss value
        loss_function = self.sparse_categorical_crossentropy

        # calculate loss with regularization if layers are provided
        if dense_layers:
            regularization_loss = 0 # initialize regularization loss
            for layer in dense_layers:
                regularization_loss += loss_function.loss_with_regularization(layer)
            avg_loss_batches = loss_function.calculate_avg_loss(self.y_preds, y_true) + regularization_loss
        else:
            avg_loss_batches = loss_function.calculate_avg_loss(self.y_preds, y_true)

        return avg_loss_batches
    
    def backward(self, y_true):
        """
        Perform the backward pass for the softmax cross-entropy loss function.

        Parameters:
            y_true (numpy.ndarray): Array of true class labels for each sample in the batch (Integer values).

        This method calculates the gradient of the loss with respect to the softmax inputs (dl_dInputs).
        """

        # make a copy of y_preds since we modify it
        self.dl_dInputs = self.y_preds.copy()

        # subtract 1 (true class label) from the predicted probability of the true class for each sample
        n_batches = len(self.y_preds) # number of samples in the batch
        self.dl_dInputs[range(n_batches), y_true] -= 1
        self.dl_dInputs /= n_batches # normalize

class Softmax_Categorical:
    def __init__(self):
        self.softmax = Softmax()
        self.categorical_crossentropy = CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        """
        Perform the forward pass through the softmax activation function and calculate the average loss.

        Parameters:
            inputs (ndarray): The input data to the softmax activation function.
            y_true (ndarray): The true labels for the input data (One-hot encoded).

        Returns:
            float: The average loss value for the batch.
        """

        # Perform the forward pass through the softmax activation function
        self.softmax.forward(inputs)
        self.y_preds = self.softmax.output

        # Calculate the loss value
        loss = self.categorical_crossentropy
        avg_loss_batches = loss.calculate_avg_loss(self.y_preds, y_true)
        return avg_loss_batches
    
    def backward(self, y_true):
        """
        Perform the backward pass for the softmax cross-entropy loss function.

        Parameters:
            y_true (numpy.ndarray): Array of true class labels for each sample in the batch (One-hot encoded).

        This method calculates the gradient of the loss with respect to the softmax inputs (dl_dInputs).
        """

        # make a copy of y_preds since we modify it
        self.dl_dInputs = self.y_preds.copy()
        # get the index of the true class label for each sample (making it a sparse representation)
        y_true = np.argmax(y_true, axis=1)

        # subtract 1 (true class label) from the predicted probability of the true class for each sample
        n_batches = len(self.y_preds)
        self.dl_dInputs[range(n_batches), y_true] -= 1
        self.dl_dInputs /= n_batches # normalize