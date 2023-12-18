import numpy
import pathlib
import autograd.numpy as np
import autograd.numpy.random as npr
from matplotlib import pyplot as plt
from functions import solve_pde_deep_neural_network_MSE

### Use the neural network:
npr.seed(15)


current_path = pathlib.Path.cwd()

figures_path = current_path / 'figures'
PNG_path = figures_path / 'PNG'
PDF_path = figures_path / 'PDF'


## Decide the vales of arguments to the function to solve
Nx = 10
Nt = 10
x = np.linspace(0, 1, Nx)
t = np.linspace(0,1,Nt)

## Set up the parameters for the network
num_hidden_neurons_list = np.array([[50, 10], [100, 10], [250, 10], [500, 10], [1000, 10], [2000, 10]])
num_iter = 200
lmb = 0.01

MSE_list_nodes = []
iters = numpy.arange(1, num_iter + 1, 1)

for num_hidden_neurons in num_hidden_neurons_list:
    P, MSEs = solve_pde_deep_neural_network_MSE(x,t, num_hidden_neurons, num_iter, lmb)

    MSE_list_nodes.append(MSEs)


#num_hidden_layers_list = [[250, 10], [250, 250, 10], [250, 250, 250, 10], [250, 250, 250, 250, 10],
#                      [250, 250, 250, 250, 250, 10]]
#            
#MSE_list_layers = []

#for num_hidden_neurons in num_hidden_layers_list:
#    P, MSEs = solve_pde_deep_neural_network_MSE(x,t, num_hidden_neurons, num_iter, lmb)
#
#    MSE_list_layers.append(MSEs)


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
plt.savefig(PNG_path / f'DNN_MSE_vs_iter_diff_nodes.png')
plt.savefig(PDF_path / f'DNN_MSE_vs_iter_diff_nodes.pdf')

#plt.figure()
#plt.style.use('ggplot')
#plt.plot(iters, MSE_list_layers[0], label = '1 hidden layer')
#plt.plot(iters, MSE_list_layers[1], label = '2 hidden layers')
#plt.plot(iters, MSE_list_layers[2], label = '3 hidden layers')
#plt.plot(iters, MSE_list_layers[3], label = '4 hidden layers')
#plt.plot(iters, MSE_list_layers[4], label = '5 hidden layers')
#plt.xlabel('$i$')
#plt.ylabel('MSE')
#plt.legend()
#plt.savefig(PNG_path / f'DNN_MSE_vs_iter_diff_layers.png')
#plt.savefig(PDF_path / f'DNN_MSE_vs_iter_diff_layers.pdf')
plt.show()
