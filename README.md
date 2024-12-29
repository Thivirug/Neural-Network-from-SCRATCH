# _**Neural Network Implementation from Scratch**_ 🧠
A Python implementation of a neural network built entirely from scratch featuring various optimization techniques, regularization methods, and activation functions.

## ***Features***

### Network Components

* Dense (Fully Connected) Layers
* ReLU Activation
* Softmax Activation
* Dropout Layer for regularization
* Multiple Loss Functions
    * Categorical Cross-Entropy
    * Sparse Categorical Cross-Entropy
  
### Optimizers
  * Basic Gradient Descent
  * Learning Rate Decay
  * Momentum
  * ADAGRAD
  * RMSProp
  * Adam

### Regularization Techniques
* L1 Regularization (weights & biases)
* L2 Regularization (weights & biases)
* Dropout

### Dataset
* Spiral Dataset for classification tasks
* Built-in dataset splitting functionality
* Data visualization capabilities

### Project Structure
```
├── Activations.py        # ReLU and Softmax activation functions
├── Dataset.py            # Spiral dataset creation and manipulation
├── History.py           # Training history tracking
├── Layers.py            # Dense and Dropout layer implementations
├── Losses.py            # Cross-entropy loss functions
├── Metrics.py           # Accuracy calculation
├── Optimizers.py        # Various optimization algorithms
├── Softmax_CrossEntropy.py # Combined Softmax activation and Cross-entropy loss
└── SpiralDataClassification.py     # Training pipeline and execution 
```

## ***Usage***
  * Basic Example

```python
# Create dataset
train_X, train_y, val_X, val_y = create_dataset(
    n_samples_per_class=1000, 
    n_classes=3
)

# Build model components
dense_layer1, relu_activation, dense_layer2, \
softmax_cross_entropy, optimizer, dropout_layer = build_model()

# Train model
history = train(train_X, train_y, dense_layer1, relu_activation, 
               dense_layer2, dropout_layer, softmax_cross_entropy, 
               optimizer, epochs=1000)

# Visualize results
plot(history)

# Validate model
validate(val_X, val_y, dense_layer1, relu_activation, 
         dense_layer2, softmax_cross_entropy)
```

## ***Training Features***
* Configurable batch size and number of epochs
* Training history tracking (loss and accuracy)
* Visualization of training metrics
* Validation set evaluation
* Various optimization strategies

## ***Advanced Features***
* Regularization Options
```python
dense_layer = Dense(
    n_inputs=2, 
    n_neurons=100,
    weight_l1_regularizer=0,
    weight_l2_regularizer=1e-4,
    bias_l1_regularizer=0,
    bias_l2_regularizer=1e-4
)
```

* Optimization Selections
```python
# Adam optimizer
optimizer = Optimizers.Optimizer_Decay_Adam(
    learning_rate=0.1, 
    decay=1e-3, 
    beta1=0.9, 
    beta2=0.999
)

# RMSProp optimizer
optimizer = Optimizers.Optimizer_Decay_RMSProp(
    learning_rate=0.01, 
    decay=1e-3, 
    epsilon=1e-7, 
    rho=0.999
)
```

## ***Requirements***
* NumPy
* Matplotlib
* NNFS (Neural Networks From Scratch)

**Contributing:**
Feel free to submit issues and enhancement requests!
