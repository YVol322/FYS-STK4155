import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Functions import Data, initialize_W_and_b, FeedForward, BackPropagation, StochasticBackPropagation

np.random.seed(2023)

x, y, X = Data()

X_train, X_test, y_train, y_test, x_train, x_test = train_test_split(X, y, x, test_size=0.2)

n_inputs, n_features = x_train.shape

n_hidden_nodes = 4 # Number of nods in hidden layers
n_hidden_layers = 2 # Number of hidden layers
n_output_nodes = 1 # Number of hidden layers

weigths, biases = initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes)

n = 10
M = 10
n_epoch = 100

t0 = 100
t1 = 10000

sigmoid = 's'
RELU = 'r'
leaky_relu = 'l'
for i in range(n):
    z_list, a_list = FeedForward(x_train, weigths, biases, RELU)

    W_list, b_list = StochasticBackPropagation(y_train, x_train, weigths, biases, M, n_epoch, t0, t1, RELU)

    print(mean_squared_error(y_train, z_list[-1]))

z_list, a_list = FeedForward(x_train, weigths, biases, RELU)

print(f"Mean Squared Error on Train Data: {mean_squared_error(y_train, z_list[-1])}")

z_list, a_list = FeedForward(x_test, weigths, biases, RELU)

print(f"Mean Squared Error on Test Data: {mean_squared_error(y_test, z_list[-1])}")