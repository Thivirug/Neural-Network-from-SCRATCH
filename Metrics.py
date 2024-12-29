import numpy as np

class Accuracy:
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

    def calculate(self):
        # if y_true is one-hot encoded
        if len(self.y_true.shape) == 2:
            self.y_true = np.argmax(self.y_true, axis=1) # convert back to single values
        
        return np.mean(self.y_pred == self.y_true)