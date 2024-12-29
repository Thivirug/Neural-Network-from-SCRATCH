import numpy as np

class Optimizer_Basic:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_params(self, layers):
        for layer in layers:
            layer.weights += -self.learning_rate * layer.dl_dw
            layer.biases += -self.learning_rate * layer.dl_db

    def __repr__(self):
        return f"Optimizer_Basic (learning_rate={self.learning_rate:.5f})"

# -------------------------------------------------------------------------- 

class Optimizer_LR_Decay:
    def __init__(self, decay, learning_rate):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.current_lr = learning_rate

    def update_params(self, layers):
        # update learning rate if decay is set
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))

        for layer in layers:
            # update weights and biases
            layer.weights += -self.current_lr * layer.dl_dw
            layer.biases += -self.current_lr * layer.dl_db

            # increment iterations
            self.iterations += 1

    def __repr__(self):
        if self.decay:
            return f"Optimizer_LR_Decay (learning_rate={self.current_lr:.5f})"
        
# ------------------------------------------------------------------------------

class Optimizer_LR_Decay_Momentum:
    def __init__(self, learning_rate, decay, momentum_factor):
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum_factor = momentum_factor
        self.iterations = 0
        self.current_lr = learning_rate

    def update_params(self, layers):
        # update learning rate if decay is set
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))

        # update weights and biases
        self.calc_updates(layers)

        # increment iterations
        self.iterations += 1

    def calc_updates(self, layers):
        for layer in layers:
            if self.momentum_factor:
                # if layer doesn't have momentum parameters, initialize them
                if not hasattr(layer, "weight_momentums"):
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    layer.bias_momentums = np.zeros_like(layer.biases)

                # calculate the momentum updates
                weight_momentum_update = -self.current_lr * layer.dl_dw + (self.momentum_factor * layer.weight_momentums)
                bias_momentum_update = -self.current_lr * layer.dl_db + (self.momentum_factor * layer.bias_momentums)

                # update the weights and biases
                layer.weights += weight_momentum_update
                layer.biases += bias_momentum_update

                # update the momentum parameters for the next iteration
                layer.weight_momentums = weight_momentum_update
                layer.bias_momentums = bias_momentum_update

            else: # if no momentum factor is set
                layer.weights += -self.current_lr * layer.dl_dw
                layer.biases += -self.current_lr * layer.dl_db

    def __repr__(self):
        if self.decay and self.momentum_factor:
            return f"Optimizer_LR_Decay_Momentum (learning_rate={self.current_lr:.5f})"

# -------------------------------------------------------------------------------------
        
class Optimizer_Decay_ADAGRAD:
    def __init__(self, learning_rate, decay, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0
        self.current_lr = learning_rate

    def update_params(self, layers):
        # update learning rate if decay is set
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))

        # update weights and biases
        self.calc_updates(layers)

        # increment iterations
        self.iterations += 1
    
    def calc_updates(self, layers):
        for layer in layers:
            # if layer does not have the cache parameters, initialize them
            if not hasattr(layer, "weight_cache"):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)

            # calculate the cache updates
            weight_cache_update = layer.dl_dw ** 2
            bias_cache_update = layer.dl_db ** 2

            # update the cache parameters
            layer.weight_cache += weight_cache_update
            layer.bias_cache += bias_cache_update

            # update the weights and biases
            layer.weights += -self.current_lr * (layer.dl_dw / np.sqrt(layer.weight_cache + self.epsilon))
            layer.biases += -self.current_lr * (layer.dl_db / np.sqrt(layer.bias_cache + self.epsilon))

    def __repr__(self):
        if self.decay:
            return f"Optimizer_Decay_ADAGRAD (learning_rate={self.current_lr:.5f})"

# -----------------------------------------------------------------------------------------------------

class Optimizer_Decay_RMSProp:
    def __init__(self, learning_rate, decay, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho
        self.iterations = 0
        self.current_lr = learning_rate

    def update_params(self, layers):
        # update learning rate if decay is set
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))

        # update weights and biases
        self.calc_updates(layers)

        # increment iterations
        self.iterations += 1

    def calc_updates(self, layers):
        for layer in layers:
            # if layer does not have the cache parameters, initialize them
            if not hasattr(layer, "weight_cache"):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)

            # calculate the cache updates and update the cache parameters
            layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * (layer.dl_dw ** 2)
            layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * (layer.dl_db ** 2)

            # update the weights and biases
            layer.weights += -self.current_lr * (layer.dl_dw / np.sqrt(layer.weight_cache + self.epsilon))
            layer.biases += -self.current_lr * (layer.dl_db / np.sqrt(layer.bias_cache + self.epsilon))

    def __repr__(self):
        if self.decay and self.rho:
            return f"Optimizer_Decay_RMSProp (learning_rate={self.current_lr:.5f})"
        
# -----------------------------------------------------------------------------------------------------

class Optimizer_Decay_Adam:
    def __init__(self, learning_rate, decay, epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.iterations = 0
        self.current_lr = learning_rate

    def update_params(self, layers):
        # update learning rate if decay is set
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))

        # update weights and biases
        self.calc_updates(layers)

        # increment iterations
        self.iterations += 1

    def calc_updates(self, layers):
        # if layer does not have the cache or momentum parameters, initialize them

        for layer in layers:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)

            # calculate the momentum updates
            weight_momentum_update = self.beta1 * layer.weight_momentums + (1-self.beta1) * layer.dl_dw
            bias_momentum_update = self.beta1 * layer.bias_momentums + (1-self.beta1) * layer.dl_db

            # calculate momentum corrections
            weight_momentum_corrected = weight_momentum_update / (1 - self.beta1 ** (self.iterations + 1))
            bias_momentum_corrected = bias_momentum_update / (1 - self.beta1 ** (self.iterations + 1))

            # calculate the cache updates
            weight_cache_update = self.beta2 * layer.weight_cache + (1-self.beta2) * (layer.dl_dw ** 2)
            bias_cache_update = self.beta2 * layer.bias_cache + (1-self.beta2) * (layer.dl_db ** 2)

            # calculate cache corrections
            weight_cache_corrected = weight_cache_update / (1 - self.beta2 ** (self.iterations + 1))
            bias_cache_corrected = bias_cache_update / (1 - self.beta2 ** (self.iterations + 1))

            # update the weights and biases
            layer.weights += -self.current_lr * (weight_momentum_corrected / np.sqrt(weight_cache_corrected + self.epsilon))
            layer.biases += -self.current_lr * (bias_momentum_corrected / np.sqrt(bias_cache_corrected + self.epsilon))

            # update the momentum and cache parameters for the next iteration
            layer.weight_momentums = weight_momentum_update
            layer.bias_momentums = bias_momentum_update
            layer.weight_cache = weight_cache_update
            layer.bias_cache = bias_cache_update

    def __repr__(self):
        if self.decay and self.beta1 and self.beta2:
            return f"Optimizer_Decay_Adam (learning_rate={self.current_lr:.5f})"