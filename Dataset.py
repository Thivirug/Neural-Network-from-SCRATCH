import numpy as np
import nnfs
import nnfs.datasets as ds
nnfs.init()
import matplotlib.pyplot as plt

class Spiral_Dataset:
    def __init__(self, samples_per_class, n_classes):
        self.X, self.y = ds.spiral_data(samples=samples_per_class, classes=n_classes)

    def __len__(self): # special method to get the length
        return len(self.X)
    
    def __getitem__(self, idx): # special method to get the item
        return self.X[idx], self.y[idx]
    
    def __repr__(self): # special method to get the representation
        return f"Spiral Dataset: \n\t{len(self.X)} samples\n\t{self.X.shape[1]} features"
    
    def scatter_plot(self):
        plt.scatter(x=self.X[:,0], y=self.X[:,1], c=self.y, cmap='brg')
        plt.show()

    
# # Create a dataset
# dataset = Spiral_Dataset(samples_per_class=100, n_classes=3)
# print(dataset) # prints out __repr__ method
# print(f"Length: {len(dataset)}") # prints out __len__ method
# print(dataset[3]) # prints out __getitem__ method for index 3
# print(dataset.__getitem__(3)) # prints out __getitem__ method for index 3
# dataset.scatter_plot() # scatter plot of the dataset
