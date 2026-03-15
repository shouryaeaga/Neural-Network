import numpy as np

class Layer:
    def forward(self, X):
        raise NotImplementedError

    def backprop(self, grad):
        raise NotImplementedError

    def set_weights(self, weights):
        raise NotImplementedError

    def set_bias(self, bias):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_size, output_size: int):
        self.activations: np.ndarray = np.random.rand(input_size)
        self.W: np.ndarray = np.random.randn(output_size, input_size) * np.sqrt(2/input_size)
        self.b: np.ndarray = np.zeros(output_size)
        self.activation_previous = None
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, activation_previous: np.ndarray) -> np.ndarray:
        self.activation_previous = activation_previous
        self.activations = np.matmul(self.W, np.transpose(activation_previous)) + np.transpose(self.b)
        return self.activations

    def set_weights(self, W):
        self.W = W

    def set_bias(self, b):
        self.b = b

    def __str__(self):
        return "Linear layer with input dims: " + str(self.input_size) + ", output dims: " + str(self.output_size)

class ReLU(Layer):
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask

    def __str__(self):
        return "ReLU layer"

class Softmax(Layer):
    def forward(self, X):
        exponents = np.exp(X)
        sum_exp = np.sum(exponents, axis=0)
        return exponents / sum_exp

class NeuralNetwork:
    def __init__(self, hidden_layer_size: int, output_size: int, input_size: int, num_hidden_layers: int, softmax: bool = False):
        self.layers: list[Layer] = []
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be greater than or equal to 1")
        self.hidden_layer_size: int = hidden_layer_size
        self.output_size: int = output_size
        self.input_size: int = input_size
        self.num_hidden_layers: int = num_hidden_layers
        self.model_parameter_size = 0
        self.softmax: bool = softmax

    def create_architecture(self):
        self.layers = [Linear(input_size=self.input_size, output_size=self.hidden_layer_size), ReLU()]
        self.model_parameter_size += self.input_size * self.hidden_layer_size + self.hidden_layer_size
        for _ in range(self.num_hidden_layers - 1):
            self.layers.append(Linear(input_size=self.hidden_layer_size, output_size=self.hidden_layer_size))
            self.model_parameter_size += self.hidden_layer_size * self.hidden_layer_size + self.hidden_layer_size
            self.layers.append(ReLU())
        self.layers.append(Linear(input_size=self.hidden_layer_size, output_size=self.output_size))

        if self.softmax:
            self.layers.append(Softmax())

        self.model_parameter_size += self.hidden_layer_size * self.output_size + self.output_size

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def __str__(self):
        result = "Neural Network\n"
        for layer in self.layers:
            result += str(layer) + "\n"
        return result

    def set_layer_weights(self, weights: np.ndarray, layer: int):
        self.layers[layer].set_weights(weights)

    def set_layer_bias(self, bias: np.ndarray, layer: int):
        self.layers[layer].set_bias(bias)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if len(X) <= 1:
            X = np.ndarray.flatten(X)
            return np.array(np.argmax(self.forward(X)))
        output = []
        for sample in X:
            sample = np.ndarray.flatten(sample)
            output.append(np.argmax(self.forward(sample)))
        return np.array(output)