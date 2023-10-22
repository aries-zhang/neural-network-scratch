import numpy as np
import matplotlib.pyplot as plt


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        # print("input is: ")
        # print(input)
        return np.dot(self.weights, input) + self.bias

    def backward(self, output_gradient, learning_rate):
        '''
        output_gradient: dE/dY
        dE/dW = dE/dY * X^T  # how much weight contributed to the error
        dE/dB = dE/dY        # how much bias contributed to the error
        dE/dX = W^T * dE/dY  # how much input contributed to the error
        '''
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient

        return np.dot(self.weights.T, output_gradient)


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_prime)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)


class Tanh(Activation):
    def __init__(self):
        super().__init__(self.tanh, self.tanh_prime)

    def tanh(self, x):
        result= np.tanh(x)
        return result

    def tanh_prime(self, x):
        return 1 - np.tanh(x)**2


class Network:
    def __init__(self, layers) -> None:
        self.layers = layers

    def mean_square_error(self, predicted, real):
        return np.mean(np.power(real - predicted, 2))

    def mean_square_error_derivative(self, predicted, real):
        return 2 * (predicted - real) / np.size(real)

    def train(self, X, Y, learning_rate, epochs):
        print("Training...")
        for epoch in range(0, epochs):
            error = 0
            # for each input in the training dataset
            for x, y in zip(X, Y):
                # forward pass for each layer
                output = x
                for layer in self.layers:
                    output = layer.forward(output)
                # calculate the error
                error += self.mean_square_error(output, y)

                # backward pass for each layer reversed
                output_gradient = self.mean_square_error_derivative(output, y)
                for layer in reversed(self.layers):
                    output_gradient = layer.backward(output_gradient, learning_rate)
            # print this epoch's error
            error /= len(X)
            print("Epoch {epoch}/{epochs}, error: {error}----".format(epoch=epoch+1, epochs=epochs, error=error))
        print("Training is done!")

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def dump(self):
        i=0
        for layer in self.layers:
            i+=1
            if i%2 == 0:
                continue
            print("Layer {i}:".format(i=i))
            print("w: {weights}".format(weights=layer.weights))
            print("b: {bias}".format(bias=layer.bias))

if __name__ == '__main__':
    network = Network([Dense(2,3), Tanh(), Dense(3,1), Tanh()])
    X = np.reshape([[0,0],[1,0],[0,1],[1,1]], (4,2,1))
    Y = np.reshape([[0],[1],[1],[0]], (4,1,1))
    network.train(X, Y, 0.1, 1000000)

    network.dump()

    i=''
    while i != 'q':
        i = input("Enter 'q' to quit, or enter two numbers separated by a comma: ")
        if i == 'q':
            break
        if ',' not in i:
            print("Invalid input")
            continue
        [a,b] = i.split(',')
        if a.isdigit() == False or b.isdigit() == False:
            print("Invalid input")
            continue
        X = np.reshape([int(a), int(b)], (2,1))
        print("Prediction:")
        print(network.predict(X))