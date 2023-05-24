#########  Acti fun
'''import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10)
# sigmoid

def sigmoid(x):
    return 1/ (1+ np.exp(-x))
plt.plot(x,sigmoid(x))
plt.title("sigmoid")
plt.show()

def relu(x):
    return np.maximum(0,x)
plt.plot(x, relu(x))
plt.title("reLu")
plt.show()

def tanh(x):
    return np.tanh(x)
plt.plot(x,tanh(x))
plt.show()

'''

####   ANdNOT

'''
import numpy as np

def neuron(inputs, weights, threshold):
    net = np.sum(inputs * weights)
    if net > threshold:
        return 1
    else:
        return 0

def andnot(x1, x2):
    inputs = np.array([x1,x2])

    weights = np.array([-1,2])

    threshold = 1

    output = neuron(inputs, weights, threshold)
    return output

print(andnot(0,0))
print(andnot(0,1))
print(andnot(1,0))
print(andnot(1,1))
'''

#########      ASCII
'''
import numpy as np

X = np.array([[48, 49, 50, 51, 52, 53, 54, 55, 56, 57],
              [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

w = np.zeros(X.shape[0])
b = 0


def activation(z):
    return 1 if z >= 0 else 0

# Training
for epoch in range (1000):
    for x,y in zip(X[0], X[1]):
        z = np.dot(w, X[:, x-48]) + b
        y_hat = activation(z)
        error = y - y_hat
        w += error * (X[:, x-48])
        b += error

# Testing
while True:
    x = input("Input a number between o and 9")
    if not x.isdigit() or int(x) < 0 or int(x) > 9:
        print("Invalid Input")
        continue
    z = np.dot(w, X[:, int(x)]) + b
    y_hat = activation(z)
    print("Even" if y_hat == 0 else "Odd")
'''


##### Perceptron

'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

np.random.seed(1)
class1 = np.random.normal(loc=(-1,-1),scale =(0.5,0.5),size=(100,2))
class2 = np.random.normal(loc=(1,1),scale =(0.5,0.5),size=(100,2))

plt.scatter(class1[:,0],class1[:,1],color = 'blue',label ='class1')
plt.scatter(class2[:,0],class2[:,1],color = 'red' ,label = 'class2')
plt.legend()
plt.show()


np.random.seed(0)
X = np.random.randn(100,2)
y = np.array([1 if x[0] + x[1] > 0 else -1 for x in X])

clf = Perceptron(max_iter=1000)
clf.fit(X,y)

# Plot the decision boundary and the data points
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
ax.set_title('Perceptron Decision Regions')
ax.set_xlabel('X1')
ax.set_ylabel('X2')

# Create a mesh grid of points and use the classifier to make predictions on them
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.5)
plt.show()'''


######  fwd & bckp
'''
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2)
        self.y_hat = self.sigmoid(self.z2)

        return self.y_hat

    def backward(self, X, y, y_hat, learning_rate):
        error = y - y_hat
        delta_output = error * self.sigmoid_derivative(y_hat)
        delta_hidden = np.dot(delta_output, self.W2.T) * self.sigmoid_derivative(self.a1)

        self.W2 += learning_rate * np.dot(self.a1.T, delta_output)
        self.W1 += learning_rate * np.dot(X.T, delta_hidden)

    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            y_hat = self.forward(X)
            self.backward(X, y, y_hat, learning_rate)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)


nn = NeuralNetwork(2, 3, 1)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn.train(X, y, epochs=10000, learning_rate=0.1)
print(nn.forward(np.array([0, 0])))
print(nn.forward(np.array([0, 1])))
print(nn.forward(np.array([1, 0])))
print(nn.forward(np.array([1, 1])))
'''


########### multiclas
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # weights and biases for the hidden layer
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))

        # weights and biases for the output layer
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def forward(self, X):
        # activation of the hidden layer
        self.hidden_activation = self.relu(np.dot(X, self.W1) + self.b1)

        # output of the neural network
        self.output_activation = self.softmax(np.dot(self.hidden_activation, self.W2) + self.b2)

        return self.output_activation

    def backward(self, X, y, output_activation, learning_rate):
        dW2 = np.dot(self.hidden_activation.T, (output_activation - y))
        db2 = np.sum(output_activation - y, axis=0, keepdims=True)

        dW1 = np.dot(X.T, np.dot(output_activation - y, self.W2.T) * (self.hidden_activation > 0))
        db1 = np.sum(np.dot(output_activation - y, self.W2.T) * (self.hidden_activation > 0), axis=0, keepdims=True)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, learning_rate, num_epochs):

        for epoch in range(num_epochs):
            # Forward
            output_activation = self.forward(X)

            # Backward
            self.backward(X, y, output_activation, learning_rate)

            # Print the loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                loss = np.mean(-np.sum(y * np.log(output_activation), axis=1))
                print("Epoch {0}/{1} - loss: {2}".format(epoch + 1, num_epochs, loss))

    def predict(self, X):
        # Compute the output of the neural network for the given input
        output_activation = self.forward(X)

        # Convert the output to a one-hot encoded vector
        y_pred = np.zeros_like(output_activation)
        # y_pred[np.arange(len(output_activation)), output_activation
        y_pred[np.arange(len(output_activation)), output_activation.argmax(1)] = 1

        return y_pred
input_size = X_train.shape[1]
hidden_size = 100
output_size = 30
learning_rate = 0.1
num_epochs = 1000

nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(X_train, np.eye(output_size)[y_train], learning_rate, num_epochs)

y_pred = nn.predict(X_test)
accuracy = np.mean(np.equal(y_pred, np.eye(output_size)[y_test]))
print("Accuracy: {0}".format(accuracy))

'''

############    XOR
'''
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


# Neural network
class XORNeuralNetwork:
    def __init__(self):
        # Initialize weights and biases with random values
        self.W1 = np.random.randn(2, 2)
        self.b1 = np.random.randn(1, 2)
        self.W2 = np.random.randn(2, 1)
        self.b2 = np.random.randn(1, 1)

    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        # Backward propagation
        delta2 = self.a2 - y
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)

            # Backward propagation
            self.backward(X, y, learning_rate)

            # Print the loss
            loss = np.mean(-y * np.log(output) - (1 - y) * np.log(1 - output))
            if epoch % 1000 == 0:
                print(f"Epoch: {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.round(self.forward(X))


# Create an instance of the XORNeuralNetwork class
nn = XORNeuralNetwork()

# Train the neural network
epochs = 10000
learning_rate = 0.1
nn.train(X, y, epochs, learning_rate)

# Test the neural network
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = nn.predict(test_input)
print("Predicted Output:")
print(predicted_output)
'''

###############      ART
'''
import numpy as np


class ART:
    def __init__(self, num_input, rho=0.5, alpha=0.1):
        self.num_input = num_input
        self.rho = rho
        self.alpha = alpha
        self.W = np.zeros((num_input,))
        self.V = self.rho * np.linalg.norm(self.W)

    def train(self, input_pattern):
        input_pattern = input_pattern / np.linalg.norm(input_pattern)
        similarity = np.dot(self.W, input_pattern)
        if similarity < self.V:
            self.W = (1 - self.alpha) * self.W + self.alpha * input_pattern
            self.V = self.rho * np.linalg.norm(self.W)

    def predict(self, input_pattern):
        input_pattern = input_pattern / np.linalg.norm(input_pattern)
        similarity = np.dot(self.W, input_pattern)
        return similarity >= self.V

# Create an ART network with 3 inputs
art = ART(num_input=3)

# Train the network on some input patterns
art.train(np.array([1, 0, 0]))
art.train(np.array([0, 1, 0]))
art.train(np.array([0, 0, 1]))

# Predict whether some input patterns are similar to existing categories
print(art.predict(np.array([0.9, 0.1, 0])))  # Output: True
print(art.predict(np.array([0.1, 0.9, 0])))  # Output: True
print(art.predict(np.array([0, 0, 1])))  # Output: True
print(art.predict(np.array([10, 20, 5])))  # Output: False
'''

#####   backprop & fedfwd
'''
import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha

        for i in range(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, display_update=100):
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in range(epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            if epoch == 0 or (epoch + 1) % display_update == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]

        for layer in range(len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)

        error = A[-1] - y
        D = [error * self.sigmoid_deriv(A[-1])]

        for layer in range(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        D = D[::-1]

        for layer in range(len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, add_bias=True):
        p = np.atleast_2d(X)

        if add_bias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in range(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, add_bias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

#  2 input nodes, 2 hidden nodes, and 1 output node
nn = NeuralNetwork([2, 2, 1])

# train the network
nn.fit(X, y, epochs=20000, display_update=1000)

# test the  network
for (x, target) in zip(X, y):
    pred = nn.predict(x)[0][0]
    print(" data={},  prediction={:.4f}".format(x, pred))
'''


