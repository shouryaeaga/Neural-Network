import numpy as np

class Layer:
    def forward(self, X):
        raise NotImplementedError

    def backprop(self, grad):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_size, output_size: int):
        self.activations: np.ndarray = np.random.rand(input_size)
        self.W: np.ndarray = np.random.rand(output_size, input_size)
        self.b: np.ndarray = np.random.rand(output_size)
        self.activation_previous = None

    def forward(self, activation_previous: np.ndarray) -> np.ndarray:
        self.activation_previous = activation_previous
        print(np.transpose(self.activation_previous).shape)
        self.activations = np.matmul(self.W, np.transpose(activation_previous)) + np.transpose(self.b)
        return self.activations

class ReLU(Layer):
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask

class NeuralNetwork:
    def __init__(self, hidden_layer_size: int, output_size: int, input_size: int, num_hidden_layers: int):
        self.layers: list[Layer] = []
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be greater than or equal to 1")
        self.hidden_layer_size: int = hidden_layer_size
        self.output_size: int = output_size
        self.input_size: int = input_size
        self.num_hidden_layers: int = num_hidden_layers

    def create_architecture(self):
        self.layers = [Linear(input_size=self.input_size, output_size=self.hidden_layer_size)]
        for _ in range(self.num_hidden_layers - 1):
            self.layers.append(Linear(input_size=self.hidden_layer_size, output_size=self.hidden_layer_size))
        self.layers.append(Linear(input_size=self.hidden_layer_size, output_size=self.output_size))

    def print_activations(self):
        for layer in self.layers:
            print(layer.activations)

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X