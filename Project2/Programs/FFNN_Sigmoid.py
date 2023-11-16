import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Functions import Data, initialize_W_and_b, FeedForward, StochasticBackPropagation, Create_dir

np.random.seed(2023)

PNG_path, PDF_path = Create_dir('FFNN')

x, y, X = Data()

X_train, X_test, y_train, y_test, x_train, x_test = train_test_split(X, y, x, test_size=0.2)

n_inputs, n_features = x_train.shape

n_hidden_nodes = 5 # Number of nods in hidden layers
n_hidden_layers = 3 # Number of hidden layers
n_layers = np.array((1, 2, 3, 4, 5, 6))
n_nodes = np.array((5, 10, 20, 30, 40, 50))
Ms = np.array((10, 20, 25, 50, 100))
n_epoch_array = np.array((2, 3, 5, 10, 20))
t0s = np.array((1, 20, 50, 70))
t1s = np.array((100, 1000, 10000, 100000))
n_output_nodes = 1 # Number of hidden layers

MSE_10 = np.zeros((n_layers.shape[0], n_layers.shape[0]))
MSE_2 = np.zeros((n_layers.shape[0], n_layers.shape[0]))
MSE_M = np.zeros((Ms.shape[0], Ms.shape[0]))
MSE_t = np.zeros((t1s.shape[0], t1s.shape[0]))

MSES = []
ns = []

n = 10
M = 5
n_epoch = 10
t0 = 100
t1 = 10000

sigmoid = 's'
RELU = 'r'
leaky_relu = 'l'

j, l = 0, 0
for n_hidden_nodes in n_nodes:
    for n_hidden_layers in n_layers:
        weigths, biases = initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes)

        for i in range(n):

            W_list, b_list = StochasticBackPropagation(y_train, x_train, weigths, biases, M, n_epoch, t0, t1, sigmoid)

            print(i, j, l)

        z_list, a_list = FeedForward(x_train, weigths, biases, sigmoid)

        MSE_10[j,l] = mean_squared_error(y_train, z_list[-1])
        j += 1
    j = 0
    l += 1




n = 2

j, l = 0, 0
for n_hidden_nodes in n_nodes:
    for n_hidden_layers in n_layers:
        weigths, biases = initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes)

        for i in range(n):

            W_list, b_list = StochasticBackPropagation(y_train, x_train, weigths, biases, M, n_epoch, t0, t1, sigmoid)

            print(i, j, l)

        z_list, a_list = FeedForward(x_train, weigths, biases, sigmoid)

        MSE_2[j,l] = mean_squared_error(y_train, z_list[-1])
        j += 1
    j = 0
    l += 1

j, l = 0, 0
for M in Ms:
    for n_epoch in n_epoch_array:
        weigths, biases = initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes)

        for i in range(n):

            W_list, b_list = StochasticBackPropagation(y_train, x_train, weigths, biases, M, n_epoch, t0, t1, sigmoid)

            print(i, j, l)

        z_list, a_list = FeedForward(x_train, weigths, biases, sigmoid)

        MSE_M[j,l] = mean_squared_error(y_train, z_list[-1])
        j += 1
    j = 0
    l += 1


j, l = 0, 0
for t0 in t0s:
    for t1 in t1s:
        weigths, biases = initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes)

        for i in range(n):

            W_list, b_list = StochasticBackPropagation(y_train, x_train, weigths, biases, M, n_epoch, t0, t1, sigmoid)

            print(i, j, l)

        z_list, a_list = FeedForward(x_train, weigths, biases, sigmoid)

        MSE_t[j,l] = mean_squared_error(y_train, z_list[-1])
        j += 1
    j = 0
    l += 1






weigths, biases = initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes)

for i in range(n):

    W_list, b_list = StochasticBackPropagation(y_train, x_train, weigths, biases, M, n_epoch, t0, t1, sigmoid)

    z_list, a_list = FeedForward(x_train, weigths, biases, sigmoid)

    MSES.append(mean_squared_error(y_train, z_list[-1]))
    ns.append(i)



plt.figure(1)
sns.heatmap(MSE_10, cmap="YlGnBu", annot=True, square=True, xticklabels = n_layers, yticklabels = n_nodes)
plt.xlabel(r"$n_{layers}$")
plt.ylabel(r'$n_{nodes}$')
#plt.savefig(PNG_path / 'Sigmoid_M=5_ep=10_it=10_t0=100_t1=10000')
#plt.savefig(PDF_path / 'Sigmoid_M=5_ep=10_it=10_t0=100_t1=10000.pdf')


plt.figure(2)
sns.heatmap(MSE_2, cmap="YlGnBu", annot=True, square=True, xticklabels = n_layers, yticklabels = n_nodes)
plt.xlabel(r"$n_{layers}$")
plt.ylabel(r'$n_{nodes}$')
#plt.savefig(PNG_path / 'Sigmoid_M=5_ep=10_it=2_t0=100_t1=10000')
#plt.savefig(PDF_path / 'Sigmoid_M=5_ep=10_it=2_t0=100_t1=10000.pdf')


plt.figure(3)
sns.heatmap(MSE_M, cmap="YlGnBu", annot=True, square=True, xticklabels = Ms, yticklabels = n_epoch_array)
plt.xlabel(r'$n_{epochs}$')
plt.ylabel(r'$M$')
#plt.savefig(PNG_path / 'Sigmoid_N=5_L=3_it=2_t0=100_t1=10000')
#plt.savefig(PDF_path / 'Sigmoid_N=5_L=3_it=2_t0=100_t1=10000.pdf')


plt.figure(4)
sns.heatmap(MSE_t, cmap="YlGnBu", annot=True, square=True, xticklabels = t0s, yticklabels = t1s)
plt.xlabel(r'$t_{0}$')
plt.ylabel(r'$t_{1}$')
#plt.savefig(PNG_path / 'Sigmoid_M=5_ep=10_N=5_L=3_it=2_')
#plt.savefig(PDF_path / 'Sigmoid_M=5_ep=10_N=5_L=3_it=2_.pdf')

plt.figure(5)
plt.style.use('ggplot')
plt.plot(ns, MSES)
plt.xlabel('Iteration')
plt.ylabel('MSE')
#plt.savefig(PNG_path / 'Sigmoid_MSE')
#plt.savefig(PDF_path / 'Sigmoid_MSE.pdf')
plt.show()
