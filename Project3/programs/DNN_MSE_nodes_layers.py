import numpy
import pathlib
import autograd.numpy as np
import autograd.numpy.random as npr
from matplotlib import pyplot as plt
from functions import solve_pde_deep_neural_network

npr.seed(15) # seed for using autograd.numpy.random.


current_path = pathlib.Path.cwd() # Currect working directory path.

# Pathes, where the figures will be saved to.
figures_path = current_path / 'figures'
PNG_path = figures_path / 'PNG'
PDF_path = figures_path / 'PDF'


# x and t arrays elements number.
Nx = 10
Nt = 10

# Creating x and t arrays.
x = np.linspace(0, 1, Nx)
t = np.linspace(0,1, Nt)

# MSE bool for the solve_pde_deep_neural_network function.
mse = 1

# DNN's parameters.
# Array with different number of nodes in the first hidden layer.
# [50, 10] - [# nodes in 1st layer, # nodes input layer].
num_hidden_neurons_list = np.array([[50, 10], [100, 10], [250, 10], [500, 10], [1000, 10], [2000, 10]])
num_iter = 250
lmb = 0.01

# Initialize list, that will contain MSE for all iteration for all architectures,
# defined in num_hidden_neurons_list.
MSE_list_nodes = []

# Iterations array.
iters = numpy.arange(1, num_iter + 1, 1)

# Train DNN and collect MSEs list for all elements in the num_hidden_neurons_list list.
for num_hidden_neurons in num_hidden_neurons_list:
    P, MSEs = solve_pde_deep_neural_network(x,t, num_hidden_neurons, num_iter, lmb, mse)

    MSE_list_nodes.append(MSEs)


# DNN's parameters.
# Array with different number of hidden layers and 250 nodes in each layer.
# [250, 10] - [# nodes in 1st layer, # nodes input layer],
# [250, 250, 10] - [# nodes in 1st layer, # nodes in 2nd layer, # nodes input layer]
num_hidden_layers_list = [[250, 10], [250, 250, 10], [250, 250, 250, 10], [250, 250, 250, 250, 10],
                      [250, 250, 250, 250, 250, 10]]
            

# Initialize list, that will contain MSE for all iteration for all architectures,
# defined in num_hidden_neurons_list.
MSE_list_layers = []

# Train DNN and collect MSEs list for all elements in the num_hidden_neurons_list list.
for num_hidden_neurons in num_hidden_layers_list:
    P, MSEs = solve_pde_deep_neural_network(x,t, num_hidden_neurons, num_iter, lmb, mse)

    MSE_list_layers.append(MSEs)




# Plot MSEs for different number of nodes and 1 layer.
plt.figure()
plt.style.use('ggplot')
plt.plot(iters, MSE_list_nodes[0], label = '50 nodes')
plt.plot(iters, MSE_list_nodes[1], label = '100 nodes')
plt.plot(iters, MSE_list_nodes[2], label = '250 nodes')
plt.plot(iters, MSE_list_nodes[3], label = '500 nodes')
plt.plot(iters, MSE_list_nodes[4], label = '1000 nodes')
plt.plot(iters, MSE_list_nodes[5], label = '2000 nodes')
plt.xlabel('$i$')
plt.ylabel('MSE')
plt.legend()
#plt.savefig(PNG_path / f'DNN_MSE_vs_iter_diff_nodes.png')
#plt.savefig(PDF_path / f'DNN_MSE_vs_iter_diff_nodes.pdf')




# Plot MSEs for different number of layes and 250 hidden nodes.
plt.figure()
plt.style.use('ggplot')
plt.plot(iters, MSE_list_layers[0], label = '1 hidden layer')
plt.plot(iters, MSE_list_layers[1], label = '2 hidden layers')
plt.plot(iters, MSE_list_layers[2], label = '3 hidden layers')
plt.plot(iters, MSE_list_layers[3], label = '4 hidden layers')
plt.plot(iters, MSE_list_layers[4], label = '5 hidden layers')
plt.xlabel('$i$')
plt.ylabel('MSE')
plt.legend()
#plt.savefig(PNG_path / f'DNN_MSE_vs_iter_diff_layers.png')
#plt.savefig(PDF_path / f'DNN_MSE_vs_iter_diff_layers.pdf')
plt.show()
