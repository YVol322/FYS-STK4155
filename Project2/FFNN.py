import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Functions import Data, sigmoid, sigmoid_derivative, Costfunction_grad
import matplotlib.pyplot as plt

np.random.seed(2023)

x, y, X, n_inputs, degree = Data()

X_train, X_test, y_train, y_test, x_train, x_test = train_test_split(X, y, x, test_size=0.2)

n_inputs, n_features = X_train.shape

gamma = 0.001

n_hidden_nodes = 4 # Number of nods in hidden layers
n_hidden_layers = 1 # Number of hidden layers
n_output_nodes = 1 # Number of hidden layers

weigths = []
biases = []

hidden_weights_1 = np.random.randn(n_features, n_hidden_nodes)
hidden_bias_1 = np.zeros(n_hidden_nodes) + 0.01

weigths.append(hidden_weights_1)
biases.append(hidden_bias_1)

for i in range(n_hidden_layers - 1):
    hidden_weights_ = np.random.randn(n_hidden_nodes, n_hidden_nodes)
    hidden_bias_ = np.zeros(n_hidden_nodes) + 0.01
    weigths.append(hidden_weights_)
    biases.append(hidden_bias_)

output_weights = np.random.randn(n_hidden_nodes, n_output_nodes)
output_bias = np.zeros(n_output_nodes) + 0.01
weigths.append(output_weights)
biases.append(output_bias)

def FeedForward(X, W_list, b_list):
    z_list = []
    a_list = []

    z_1 = X @ W_list[0] + b_list[0]
    z_list.append(z_1)

    a_1 = sigmoid(z_1)
    a_list.append(a_1)

    for i in range(len(W_list) - 1):
        z_i = a_list[i] @ W_list[i+1] + b_list[i+1]
        z_list.append(z_i)

        if i == len(W_list) - 2: break

        a_i = sigmoid(z_i)
        a_list.append(a_i)
    
    return z_list, a_list

def BackPropagation(y_train, X_train, W_list, b_list, a_list, z_list):
    delta_list = []
    delta_out = Costfunction_grad(y_train, z_list[-1])
    delta_list.append(delta_out)

    for i in range(len(W_list) - 1):
        delta_i = (delta_list[-1] @ (W_list[-1 - i]).T) * sigmoid_derivative(a_list[-1 - i])
        delta_list.append(delta_i)
    
    delta_list.reverse()

    W_list[0] -= gamma * (X_train.T @ delta_list[0])
    b_list[0] -= gamma * np.sum(delta_list[0])

    for i in range(len(W_list) - 1):
        W_list[i + 1] -= gamma * (a_list[i].T @ delta_list[i + 1])
        b_list[i + 1] -= gamma * np.sum(delta_list[i + 1])


    return W_list, b_list



n = 10000

for i in range(n):
    z_list, a_list = FeedForward(X_train, weigths, biases)

    W_list, b_list = BackPropagation(y_train, X_train, weigths, biases, a_list, z_list)

    if(mean_squared_error(y_train, z_list[-1]) < 1e-4): break

print(f"Mean Squared Error on Train Data: {mean_squared_error(y_train, z_list[-1])}")

z_list, a_list = FeedForward(X_test, weigths, biases)

print(f"Mean Squared Error on Test Data: {mean_squared_error(y_test, z_list[-1])}")