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
Nx = 10; Nt = 10
x = np.linspace(0, 1, Nx)
t = np.linspace(0,1,Nt)

## Set up the parameters for the network
num_hidden_neurons = [20, 10]
num_iter = 200
lmb = 0.01

iters = numpy.arange(1, num_iter + 1, 1)

P, MSEs = solve_pde_deep_neural_network_MSE(x,t, num_hidden_neurons, num_iter, lmb)


plt.figure()
plt.style.use('ggplot')
plt.plot(iters[5:], MSEs[5:])
plt.xlabel('$i$')
plt.ylabel('MSE')
plt.savefig(PNG_path / f'DNN_MSE_vs_iter_relu.png')
plt.savefig(PDF_path / f'DNN_MSE_vs_iter_relu.pdf')
plt.show()
