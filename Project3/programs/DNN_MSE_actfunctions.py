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
#num_hidden_neurons = [100, 10] # uncomment this and line 125 in the functions.py to use tanh.
#num_hidden_neurons = [20, 10] # uncomment this and line 126 in the functions.py to use RELU.
num_iter = 200
lmb = 0.01

# MSE bool for the solve_pde_deep_neural_network function.
mse = 1

# Iterations array.
iters = numpy.arange(1, num_iter + 1, 1)

# Train DNN and collec MSE array.
P, MSEs = solve_pde_deep_neural_network(x,t, num_hidden_neurons, num_iter, lmb, mse)




# Plot MSE vs current iteration number.
plt.figure()
plt.style.use('ggplot')
plt.plot(iters[5:], MSEs[5:])
plt.xlabel('$i$')
plt.ylabel('MSE')
#plt.savefig(PNG_path / f'DNN_MSE_vs_iter_sigmoid.png')
#plt.savefig(PDF_path / f'DNN_MSE_vs_iter_sigmoid.pdf')
#plt.savefig(PNG_path / f'DNN_MSE_vs_iter_tanh.png') # Uncomment this line if you're using tanh
#plt.savefig(PDF_path / f'DNN_MSE_vs_iter_tanh.pdf') # Uncomment this line if you're using tanh
#plt.savefig(PNG_path / f'DNN_MSE_vs_iter_relu.png') # Uncomment this line if you're using relu
#plt.savefig(PDF_path / f'DNN_MSE_vs_iter_relu.pdf') # Uncomment this line if you're using relu
plt.show()
