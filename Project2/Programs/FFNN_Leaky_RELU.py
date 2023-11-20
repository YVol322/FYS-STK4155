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

n_hidden_nodes = 15 # Default number of nods in hidden layers
n_hidden_layers = 1 # Default number of hidden layers
n_output_nodes = 1 # Default number of hidden layers
n = 5 # Default number of iterations.
M = 5 # Default mini-batch size.
n_epoch = 10 # Default number of epochs.
t0 = 10 # Defaut t_0 parameter.
t1 = 1000000 # Default t_1 parameter.

n_layers = np.array((1, 2, 3)) # Array of number of layers.
n_nodes = np.array((5, 10, 15)) # Array of number of nodes.


# Initialising empty matrices.
MSE_5 = np.zeros((n_layers.shape[0], n_layers.shape[0]))
MSE_2 = np.zeros((n_layers.shape[0], n_layers.shape[0]))


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
            W_list, b_list = StochasticBackPropagation(y_train, x_train, weigths, biases, M, n_epoch, t0, t1, leaky_relu)

            print(i, j, l)

        # Running Feed Forward step to make a prediction.
        z_list, a_list = FeedForward(x_train, weigths, biases, leaky_relu)

        # Adding MSEs to the matrix.
        MSE_5[j,l] = mean_squared_error(y_train, z_list[-1])
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
            W_list, b_list = StochasticBackPropagation(y_train, x_train, weigths, biases, M, n_epoch, t0, t1, RELU)

            print(i, j, l)

        # Running Feed Forward step to make a prediction.
        z_list, a_list = FeedForward(x_train, weigths, biases, RELU)

        # Adding MSEs to the matrix.
        MSE_2[j,l] = mean_squared_error(y_train, z_list[-1])
        j += 1
    j = 0
    l += 1



# Initilising weights and biases lists.
weigths, biases = initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes)

n = 10 # Setting bigger number of iterations.

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
sns.heatmap(MSE_5, cmap="YlGnBu", annot=True, square=True, xticklabels = n_layers, yticklabels = n_nodes)
plt.xlabel(r"$n_{layers}$")
plt.ylabel(r'$n_{nodes}$')
#plt.savefig(PNG_path / 'leakyRELU_M=5_ep=10_it=5_t0=100_t1=10000')
#plt.savefig(PDF_path / 'leakyRELU_M=5_ep=10_it=5_t0=100_t1=10000.pdf')


plt.figure(2)
sns.heatmap(MSE_2, cmap="YlGnBu", annot=True, square=True, xticklabels = n_layers, yticklabels = n_nodes)
plt.xlabel(r"$n_{layers}$")
plt.ylabel(r'$n_{nodes}$')
#plt.savefig(PNG_path / 'leakyRELU_M=5_ep=10_it=2_t0=100_t1=10000')
#plt.savefig(PDF_path / 'leakyRELU_M=5_ep=10_it=2_t0=100_t1=10000.pdf')

plt.figure(5)
plt.style.use('ggplot')
plt.plot(ns, MSES)
plt.xlabel('Iteration')
plt.ylabel('MSE')
#plt.savefig(PNG_path / 'leakyRELU_MSE')
#plt.savefig(PDF_path / 'leakyRELU_MSE.pdf')
plt.show()