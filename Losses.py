import numpy as np

# Parent class for all loss functions
class Loss:
    def calculate_avg_loss(self, y_preds, y_true):
        """
        Calculates the average loss value for a batch of predictions.

        Parameters:
            y_preds (numpy.ndarray): Predicted probabilities for each class. Shape (n_samples, n_classes).
            y_true (numpy.ndarray): True class labels. 

        Returns:
            float: Average loss value.
        """

        sample_losses = self.forward(y_preds, y_true)

        # calculate the average loss with regularization
        avg_loss = np.mean(sample_losses)
        return avg_loss
    
    def loss_with_regularization(self, layer):
        """
        Calculates the loss value with regularization.

        Parameters:
            layer (object): The layer object.

        Returns:
            float: Loss value with regularization.
        """

        regularization_loss = 0 # initialize regularization loss

        # L1 regularization on weights
        if layer.weight_l1_regularizer > 0:
            regularization_loss += layer.weight_l1_regularizer * np.sum(np.abs(layer.weights))

        # L2 regularization on weights
        if layer.weight_l2_regularizer > 0:
            regularization_loss += layer.weight_l2_regularizer * np.sum(np.square(layer.weights))

        # L1 regularization on biases
        if layer.bias_l1_regularizer > 0:
            regularization_loss += layer.bias_l1_regularizer * np.sum(np.abs(layer.biases))

        # L2 regularization on biases
        if layer.bias_l2_regularizer > 0:
            regularization_loss += layer.bias_l2_regularizer * np.sum(np.square(layer.biases))
        
        return regularization_loss
        

# Child classes

# 1) Categorical Cross-Entropy loss function
class CategoricalCrossEntropy(Loss):
    """
    Implements the Categorical Cross-Entropy loss function (For one-hot encoded labels).
    """

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def forward(self, y_preds, y_true):
        """
        Computes the negative log likelihood loss.

        Parameters:
            y_preds (numpy.ndarray): Predicted probabilities for each class. Shape (n_samples, n_classes).
            y_true (numpy.ndarray): One-hot encoded true class labels. Shape (n_samples, n_classes).

        Returns:
            numpy.ndarray: Negative log likelihood loss for each sample.
        """
        
        # clip y_pred to prevent log(0) and log(1)
        y_preds = np.clip(y_preds, self.epsilon, 1 - self.epsilon)

        # element-wise multiplication of y_pred & y_true
        element_mult = np.multiply(y_true, y_preds)

        # sum over rows to get the predicted probability of the true class
        pred_probs_true_class = np.sum(element_mult, axis=1)

        # negative log likelihoods
        true_class_value = 1
        negative_log_likelihoods = -true_class_value * np.log(pred_probs_true_class)
        return negative_log_likelihoods

# 2) Sparse Categorical Cross-Entropy loss function
class SparseCategoricalCrossEntropy(Loss):
    """
    Implements the Sparse Categorical Cross-Entropy loss function (For integer class labels).
    """

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        
    def forward(self, y_preds, y_true):
        """
        Computes the negative log likelihood loss for a set of predictions and true labels.

        Parameters:
            y_preds (numpy.ndarray): Predicted probabilities for each class, shape (n_samples, n_classes).
            y_true (numpy.ndarray): True class labels, shape (n_samples,).

        Returns:
            numpy.ndarray: Negative log likelihood loss for each sample, shape (n_samples,).
        """

        # clip y_pred to prevent log(0) and log(1)
        y_preds = np.clip(y_preds, self.epsilon, 1 - self.epsilon)

        # get the predicted probabilities for the target classes
        pred_probs_true_class = y_preds[range(len(y_preds)), y_true]

        # negative log likelihoods
        true_class_value = 1
        negative_log_likelihoods = -true_class_value * np.log(pred_probs_true_class)
        return negative_log_likelihoods