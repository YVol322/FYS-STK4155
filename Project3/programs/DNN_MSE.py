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

# DNN's parameters.
# First element in the list is the number of hidden layer nodes, second argument is the number of nodes
# in the output layer (need to be equal to Nx).
num_hidden_neurons = [250, 10]
num_iter = 1000
lmb = 0.01

# MSE bool for the solve_pde_deep_neural_network function.
mse = 1

# Train DNN and collec MSE array.
P, MSEs = solve_pde_deep_neural_network(x,t, num_hidden_neurons, num_iter, lmb, mse)

# Iterations array.
iters = numpy.arange(1, num_iter + 1, 1)




# Plot MSE vs current iteration number.
plt.figure()
plt.style.use('ggplot')
plt.plot(iters, MSEs)
plt.xlabel('$i$')
plt.ylabel('MSE')
#plt.savefig(PNG_path / f'DNN_MSE_vs_iter_1layer_250nodes_sigmoid.png')
#plt.savefig(PDF_path / f'DNN_MSE_vs_iter_1layer_250nodes_sigmoid.pdf')
plt.show()
