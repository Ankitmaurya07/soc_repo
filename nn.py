import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

def ReLU(Z):
    return np.maximum(0, Z)

def DERIVATIVE_RELU(Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)

class NN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize basic stats of NN
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

        # Initialize activations and gradients
        self.A0 = None
        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None
        self.dW2 = None
        self.db2 = None
        self.dW1 = None
        self.db1 = None

    # Do the forward pass and evaluate the values of A0, Z1, A1, Z2, A2
    def forward_propagation(self, X):
        self.A0 = X
        self.Z1 = np.dot(self.W1, self.A0) + self.b1
        self.A1 = ReLU(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = softmax(self.Z2)

    # Convert the input y into a one hot encoded array.
    def one_hot(self, y, num_classes):
        one_hot_y = np.zeros((num_classes, y.size))
        one_hot_y[y, np.arange(y.size)] = 1
        return one_hot_y

    # Calculate the derivative of the loss function with respect to W2, b2, W1, b1 in dW2, db2, dW1, db1 respectively
    def backward_propagation(self, X, y):
        m = X.shape[1]
        one_hot_y = self.one_hot(y, self.output_size)
        
        dZ2 = self.A2 - one_hot_y
        self.dW2 = 1 / m * np.dot(dZ2, self.A1.T)
        self.db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(self.W2.T, dZ2) * DERIVATIVE_RELU(self.Z1)
        self.dW1 = 1 / m * np.dot(dZ1, self.A0.T)
        self.db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    
    # Update the parameters W1, W2, b1, b2
    def update_params(self):
        self.W1 -= self.learning_rate * self.dW1
        self.b1 -= self.learning_rate * self.db1
        self.W2 -= self.learning_rate * self.dW2
        self.b2 -= self.learning_rate * self.db2
    
    # Get the predictions for the dataset
    def get_predictions(self, X):
        self.forward_propagation(X)
        return np.argmax(self.A2, axis=0)

    # Get the accuracy of the model on the dataset
    def get_accuracy(self, X, y):
        predictions = self.get_predictions(X)
        return np.mean(predictions == y)

    # Run gradient descent on the model to get the values of the parameters
    def gradient_descent(self, X, y, iters=1000):
        for i in range(iters):
            self.forward_propagation(X)
            self.backward_propagation(X, y)
            self.update_params()
            if i % 100 == 0:
                cost = self.cross_entropy_loss(X, y)
                accuracy = self.get_accuracy(X, y)
                print(f"Iteration {i}: Cost {cost}, Accuracy {accuracy}")
    
    # Evaluate loss using cross-entropy-loss formula.
    def cross_entropy_loss(self, X, y):
        m = X.shape[1]
        one_hot_y = self.one_hot(y, self.output_size)
        loss = -np.sum(one_hot_y * np.log(self.A2 + 1e-8)) / m
        return loss

    # Let me help a bit hehe :)
    def show_predictions(self, X, y, num_samples=10):
        random_indices = np.random.randint(0, X.shape[1], size=num_samples)

        for index in random_indices:
            sample_image = X[:, index].reshape((28, 28))
            plt.imshow(sample_image, cmap='gray')
            plt.title(f"Actual: {y[index]}, Predicted: {self.get_predictions(X)[index]}")
            plt.show()

# Load and preprocess the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalize the data
miu = np.mean(X_train, axis=(0, 1), keepdims=True)
stds = np.std(X_train, axis=(0, 1), keepdims=True)
X_normal_train = (X_train - miu) / (stds + 1e-7)

mius = np.mean(X_test, axis=(0, 1), keepdims=True)
stdse = np.std(X_test, axis=(0, 1), keepdims=True)
X_normal_test = (X_test - mius) / (stdse + 1e-7)

# Reshape the data
X_normal_train = X_normal_train.reshape((60000, -1)).T
X_normal_test = X_normal_test.reshape((10000, -1)).T

# Initialize and train the neural network
nn = NN(input_size=784, hidden_size=128, output_size=10, learning_rate=0.01)
nn.gradient_descent(X_normal_train, Y_train, iters=1000)

# Evaluate the model
train_accuracy = nn.get_accuracy(X_normal_train, Y_train)
test_accuracy = nn.get_accuracy(X_normal_test, Y_test)

print(f"Training accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")

# Show predictions
nn.show_predictions(X_normal_test, Y_test, num_samples=10)
