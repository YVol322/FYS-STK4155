import numpy
import pathlib
import autograd.numpy as np
import autograd.numpy.random as npr
from matplotlib import pyplot as plt
from functions import solve_pde_deep_neural_network

### Use the neural network:
npr.seed(15)


current_path = pathlib.Path.cwd()

figures_path = current_path / 'figures'
PNG_path = figures_path / 'PNG'
PDF_path = figures_path / 'PDF'


## Decide the vales of arguments to the function to solve
Nx = 10; Nt = 10
x = np.linspace(0, 1, Nx)
t = np.linspace(0,1,Nt)

## Set up the parameters for the network
num_hidden_neurons = [250, 10]
num_iter = 1000
lmb = 0.01


P, MSEs = solve_pde_deep_neural_network(x,t, num_hidden_neurons, num_iter, lmb)

iters = numpy.arange(1, num_iter + 1, 1)

plt.figure()
plt.style.use('ggplot')
plt.plot(iters, MSEs)
plt.xlabel('$i$')
plt.ylabel('MSE')
plt.savefig(PNG_path / f'DNN_MSE_vs_iter_1layer_250nodes_sigmoid.png')
plt.savefig(PDF_path / f'DNN_MSE_vs_iter_1layer_250nodes_sigmoid.pdf')
plt.show()
