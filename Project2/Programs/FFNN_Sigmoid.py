import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Functions import Data, initialize_W_and_b, FeedForward, StochasticBackPropagation, Create_dir

np.random.seed(2023) # Setting the seed results can be repoduced.

PNG_path, PDF_path = Create_dir('FFNN') # Creating directories to save figures to.

x, y, X = Data() # Generating data.
X_train, X_test, y_train, y_test, x_train, x_test = train_test_split(X, y, x, test_size=0.2) # Splitting data.
n_inputs, n_features = x_train.shape # Collecting number of data-points and features of x array.

n_hidden_nodes = 5 # Default number of nods in hidden layers
n_hidden_layers = 3 # Default number of hidden layers
n_output_nodes = 1 # Default number of hidden layers
n = 10 # Default number of iterations.
M = 5 # Default mini-batch size.
n_epoch = 10 # Default number of epochs.
t0 = 100 # Defaut t_0 parameter.
t1 = 10000 # Default t_1 parameter.


n_layers = np.array((1, 2, 3, 4, 5, 6)) # Array of number of layers.
n_nodes = np.array((5, 10, 20, 30, 40, 50)) # Array of number of nodes.
Ms = np.array((10, 20, 25, 50, 100)) # Array of mini-batch sizes.
n_epoch_array = np.array((2, 3, 5, 10, 20)) # Array of number of epochs.
t0s = np.array((1, 20, 50, 70)) # Array of t_0 parameters.
t1s = np.array((100, 1000, 10000, 100000)) # Array of t_1 parameters.


# Initialising empty matrices.
MSE_10 = np.zeros((n_layers.shape[0], n_layers.shape[0]))
MSE_2 = np.zeros((n_layers.shape[0], n_layers.shape[0]))
MSE_M = np.zeros((Ms.shape[0], Ms.shape[0]))
MSE_t = np.zeros((t1s.shape[0], t1s.shape[0]))

# Initialising empty lists.
MSES = []
ns = []

# String for activation function choise.
sigmoid = 's'
RELU = 'r'
leaky_relu = 'l'

j, l = 0, 0 # Initial indices.

# Loop over nodes number.
for n_hidden_nodes in n_nodes:

    # Loop over layers number.
    for n_hidden_layers in n_layers:

        # Initilising weights and biases lists.
        weigths, biases = initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes)

        # Loop over number of iterations.
        for i in range(n):
            
            # Updating weights and biases with Back Prop algo.
            W_list, b_list = StochasticBackPropagation(y_train, x_train, weigths, biases, M, n_epoch, t0, t1, sigmoid)

            print(i, j, l)

        # Running Feed Forward step to make a prediction.
        z_list, a_list = FeedForward(x_train, weigths, biases, sigmoid)

        # Adding MSEs to the matrix.
        MSE_10[j,l] = mean_squared_error(y_train, z_list[-1])
        j += 1
    j = 0
    l += 1




n = 2 # Setting smaller number of iterations.

j, l = 0, 0 # Initial indices.

# Loop over nodes number.
for n_hidden_nodes in n_nodes:

    # Loop over layers number.
    for n_hidden_layers in n_layers:

        # Initilising weights and biases lists.
        weigths, biases = initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes)

        # Loop over number of iterations.
        for i in range(n):
            
            # Updating weights and biases with Back Prop algo.
            W_list, b_list = StochasticBackPropagation(y_train, x_train, weigths, biases, M, n_epoch, t0, t1, sigmoid)

            print(i, j, l)

        # Running Feed Forward step to make a prediction.
        z_list, a_list = FeedForward(x_train, weigths, biases, sigmoid)

        # Adding MSEs to the matrix.
        MSE_2[j,l] = mean_squared_error(y_train, z_list[-1])
        j += 1
    j = 0
    l += 1



j, l = 0, 0 # Initial indices.

# Loop over mini-batch size.
for M in Ms:

    # Loop over number of epochs.
    for n_epoch in n_epoch_array:

        # Initilising weights and biases lists.
        weigths, biases = initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes)

        # Loop over number of iterations.
        for i in range(n):

            # Updating weights and biases with Back Prop algo.
            W_list, b_list = StochasticBackPropagation(y_train, x_train, weigths, biases, M, n_epoch, t0, t1, sigmoid)

            print(i, j, l)

        # Running Feed Forward step to make a prediction.
        z_list, a_list = FeedForward(x_train, weigths, biases, sigmoid)

        # Adding MSEs to the matrix.
        MSE_M[j,l] = mean_squared_error(y_train, z_list[-1])
        j += 1
    j = 0
    l += 1


j, l = 0, 0 # Initial indices.

# Loop over t_0 parameters.
for t0 in t0s:

    # Loop over t_1 parameters.
    for t1 in t1s:

        # Initilising weights and biases lists.
        weigths, biases = initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes)

        # Loop over number of iterations.
        for i in range(n):

            # Updating weights and biases with Back Prop algo.
            W_list, b_list = StochasticBackPropagation(y_train, x_train, weigths, biases, M, n_epoch, t0, t1, sigmoid)

            print(i, j, l)

        # Running Feed Forward step to make a prediction.
        z_list, a_list = FeedForward(x_train, weigths, biases, sigmoid)

        # Adding MSEs to the matrix.
        MSE_t[j,l] = mean_squared_error(y_train, z_list[-1])
        j += 1
    j = 0
    l += 1





# Initilising weights and biases lists.
weigths, biases = initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes)

# Loop over number of iterations.
for i in range(n):

    # Updating weights and biases with Back Prop algo.
    W_list, b_list = StochasticBackPropagation(y_train, x_train, weigths, biases, M, n_epoch, t0, t1, sigmoid)

    # Running Feed Forward step to make a prediction.
    z_list, a_list = FeedForward(x_train, weigths, biases, sigmoid)

    MSES.append(mean_squared_error(y_train, z_list[-1])) # Adding MSE to the list.
    ns.append(i) # Adding iteration number to the list.


# Generating some plots.

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
