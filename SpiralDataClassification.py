from Dataset import Spiral_Dataset
from Layers import Dense, Dropout
from Activations import ReLU
from Softmax_CrossEntropy import Softmax_Sparse
import Optimizers
from Metrics import Accuracy
import numpy as np
import History
import matplotlib.pyplot as plt

def create_dataset(n_samples_per_class, n_classes, val_size=0.2):
    # create the spiral dataset
    dataset = Spiral_Dataset(samples_per_class=n_samples_per_class, n_classes=n_classes)
    X, y = dataset.X, dataset.y

    # shuffle the dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # split the dataset into training and testing sets
    train_split = int(len(X) * (1 - val_size))
    train_X, train_y = X[:train_split, :], y[:train_split] 
    val_X, val_y = X[train_split:, :], y[train_split:]

    print(f"Training Set: {len(train_X)} samples\nTesting Set: {len(val_X)} samples\n")

    return train_X, train_y, val_X, val_y

def build_model():
    dense_layer1 = Dense(
        n_inputs=2, 
        n_neurons=100,
        weight_l2_regularizer=1e-4,
        bias_l2_regularizer=1e-4
    ) # 2 input features, n_neurons output neurons, & L2 regularization 

    relu_activation = ReLU() # ReLU activation (to be used with the dense layer)

    dropout_layer = Dropout(dropout_rate=0.2) # dropout layer with a dropout rate of 0.2

    dense_layer2 = Dense(
        n_inputs=100, 
        n_neurons=3
    ) # n_neurons of previous layer inputs, 3 output neurons (for 3 classes)

    softmax_cross_entropy = Softmax_Sparse() # combined Softmax activation and cross-entropy loss

    #optimizer = Optimizers.Optimizer_Basic(learning_rate=1) # basic gradient descent optimizer 
    #optimizer = Optimizers.Optimizer_LR_Decay(decay=1e-3, learning_rate=1) # learning rate decay optimizer 
    #optimizer = Optimizers.Optimizer_LR_Decay_Momentum(learning_rate=1, decay=1e-4, momentum_factor=0.7) # learning rate decay with momentum optimizer
    #optimizer = Optimizers.Optimizer_Decay_ADAGRAD(learning_rate=1, decay=1e-4, epsilon=1e-7) # ADAGRAD optimizer
    #optimizer = Optimizers.Optimizer_Decay_RMSProp(learning_rate=0.01, decay=1e-3, epsilon=1e-7, rho=0.999) # RMSProp optimizer
    optimizer = Optimizers.Optimizer_Decay_Adam(learning_rate=0.1, decay=1e-3, beta1=0.9, beta2=0.999) # Adam optimizer

    return dense_layer1, relu_activation, dense_layer2, softmax_cross_entropy, optimizer, dropout_layer

def forward_pass(inputs, dense_layer1, relu_activation, dense_layer2, softmax_cross_entropy, y_true, dropout_layer=None, test=False):
    # forward pass through the 1st dense layer
    dense_layer1.forward(inputs)

    # forward pass through the activation function
    relu_activation.forward(dense_layer1.output)

    # forward pass through the dropout layer
    if dropout_layer:
        dropout_layer.forward(relu_activation.output)
        # forward pass through the 2nd dense layer
        dense_layer2.forward(dropout_layer.output)
    else:
        # forward pass through the 2nd dense layer
        dense_layer2.forward(relu_activation.output)
    
    # forward pass through the combined Softmax activation and cross-entropy loss
    if test: 
        loss = softmax_cross_entropy.forward(dense_layer2.output, y_true)
    else: 
        loss = softmax_cross_entropy.forward(dense_layer2.output, y_true, dense_layers=[dense_layer1, dense_layer2])

    return loss

def backward_pass(y_true, dense_layer1, relu_activation, dense_layer2, softmax_cross_entropy, dropout_layer):
    # backward pass through the combined Softmax activation and cross-entropy loss
    softmax_cross_entropy.backward(y_true)
    # backward pass through the 2nd dense layer
    dense_layer2.backward(softmax_cross_entropy.dl_dInputs)
    # backward pass through the dropout layer
    dropout_layer.backward(dense_layer2.dl_dX)
    # backward pass through the ReLU activation function
    relu_activation.backward(dropout_layer.dl_dX)
    # backward pass through the 1st dense layer
    dense_layer1.backward(relu_activation.dl_dInputs)

def train(X, y, dense_layer1, relu_activation, dense_layer2, dropout_layer, softmax_cross_entropy, optimizer, epochs):
    # create a history object to store the loss and accuracy values
    history = History.History()

    for epoch in range(epochs):
        # forward pass
        loss = forward_pass(X, dense_layer1, relu_activation, dense_layer2, softmax_cross_entropy, y, dropout_layer)

        # calculate accuracy
        y_preds = np.argmax(softmax_cross_entropy.y_preds, axis=1) # get the indices of the highest probability for each sample/batch
        accuracy = Accuracy(y_preds, y).calculate()

        frequency = epochs // 20
        if epoch % frequency == 0: # print the accuracy and loss every frequency epochs
            print(f"Epoch: {epoch}\n\t{optimizer}\n Accuracy: {accuracy:.3f}, Loss: {loss:.3f}")

        # backward pass
        backward_pass(y, dense_layer1, relu_activation, dense_layer2, softmax_cross_entropy, dropout_layer)

        # update weights and biases
        optimizer.update_params(layers=[dense_layer1,dense_layer2])

        # store the loss and accuracy values
        history.append(loss, accuracy)

    print("\nTraining Complete.\n")
    return history

def plot(history):
    loss = [value[0] for value in history.history] # get the loss values
    accuracy = [value[1] for value in history.history] # get the accuracy values
    epochs = range(len(loss))
    
    # create subplots
    plt.subplots(1, 2) # 1 row, 2 columns
    # common title for the subplots
    plt.suptitle("Training Loss and Accuracy")

    # plot the loss 
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st subplot
    plt.plot(epochs, loss, label="Loss", color="blue")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # plot the accuracy
    plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd subplot
    plt.plot(epochs, accuracy, label="Accuracy", color="red")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

def validate(val_X, val_y, dense_layer1, relu_activation, dense_layer2, softmax_cross_entropy):
    # forward pass
    test_loss = forward_pass(val_X, dense_layer1, relu_activation, dense_layer2, softmax_cross_entropy, val_y, test=True)

    # calculate accuracy
    y_preds = np.argmax(softmax_cross_entropy.y_preds, axis=1) # get the indices of the highest probability for each sample/batch
    accuracy = Accuracy(y_preds, val_y).calculate()

    print(f"Testing Set\n Accuracy: {accuracy:.3f}, Loss: {test_loss:.3f}")

def main():
    train_X, train_y, val_X, val_y = create_dataset(n_samples_per_class=1000, n_classes=3)
    dense_layer1, relu_activation, dense_layer2, softmax_cross_entropy, optimizer, dropout_layer = build_model()
    history = train(train_X, train_y, dense_layer1, relu_activation, dense_layer2, dropout_layer, softmax_cross_entropy, optimizer, epochs=1000)
    plot(history)
    validate(val_X, val_y, dense_layer1, relu_activation, dense_layer2, softmax_cross_entropy)

if __name__ == "__main__":
    main()