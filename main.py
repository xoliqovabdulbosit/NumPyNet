import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output, learning_rate):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        limit = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size) * limit
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, gradient_output, learning_rate):
        weights_gradient = np.dot(self.input.T, gradient_output)
        bias_gradient = np.sum(gradient_output, axis=0, keepdims=True)
        input_gradient = np.dot(gradient_output, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, gradient_output, learning_rate):
        return gradient_output * self.activation_prime(self.input)


class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid, sigmoid_prime)


class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)


class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: (x > 0).astype(float)
        super().__init__(relu, relu_prime)


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, X, Y, epochs, learning_rate, loss_function, loss_prime):
        for epoch in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                # Forward
                output = self.predict(x.reshape(1, -1))
                error += loss_function(y, output)

                # Backward
                grad = loss_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {error:.6f}")


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)
