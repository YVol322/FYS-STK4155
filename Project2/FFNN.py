import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Functions import Data, sigmoid, sigmoid_derivative
import matplotlib.pyplot as plt

np.random.seed(2023)

x, y, X, n_inputs, degree = Data()

X_train, X_test, y_train, y_test, x_train, x_test = train_test_split(X, y, x, test_size=0.2)

n_inputs, n_features = X_train.shape

gamma = 0.001

n_hidden_nodes = 4 # Number of nods in hidden layers
n_hidden_layers = 1 # Number of hidden layers
n_output_nodes = 1 # Number of hidden layers

hidden_weights = np.random.randn(n_features, n_hidden_nodes)
hidden_bias = np.zeros(n_hidden_nodes) + 0.01


#hidden_weights_2 = np.random.randn(n_features, n_hidden_nodes)
#hidden_bias_2 = np.zeros(n_hidden_nodes) + 0.01

output_weights = np.random.randn(n_hidden_nodes, n_output_nodes)
output_bias = np.zeros(n_output_nodes) + 0.01

def FeedForward(X, W1, b1, Wout, bout):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ Wout + bout

    return z2, z1, a1



def Costfunction_grad(y_true, y_pred):
    return (y_pred - y_true)

def BackPropagation(y_train, X_train, zout, W1, Wout, b1, bout, a1):
    delta_out = Costfunction_grad(y_train, zout)

    delta_hidden = (delta_out @ Wout.T) * sigmoid_derivative(a1)

    W1 -= gamma * (X_train.T @ delta_hidden)
    b1 -= gamma * np.sum(delta_hidden)
    Wout -= gamma * (a1.T @ delta_out)
    bout -= gamma * np.sum(delta_out)

    return W1, b1, Wout, bout


n = 10000

for i in range(n):
    print(i)
    z2, z1, a1 = FeedForward(X_train, hidden_weights, hidden_bias, output_weights, output_bias)

    hidden_weights, hidden_bias, output_weights, output_bias = BackPropagation(y_train, X_train, 
                                z2, hidden_weights, output_weights, hidden_bias, output_bias, a1)
    
mse_train = mean_squared_error(y_train, z2)
print(f"Mean Squared Error on Train Data: {mse_train}")

def predict(X, W1, b1, Wout, bout):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ Wout + bout
    predictions = z2

    return predictions

# Using the trained weights and biases for prediction
predictions_test = predict(X_test, hidden_weights, hidden_bias, output_weights, output_bias)

# Calculate mean squared error on test data
mse_test = mean_squared_error(y_test, predictions_test)
print(f"Mean Squared Error on Test Data: {mse_test}")