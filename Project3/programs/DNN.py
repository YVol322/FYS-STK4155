import numpy
import pathlib
from matplotlib import cm
import autograd.numpy as np
import autograd.numpy.random as npr
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from functions import solve_pde_deep_neural_network, g_trial, g_analytic


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
mse = 0

# Train DNN.
P = solve_pde_deep_neural_network(x,t, num_hidden_neurons, num_iter, lmb, mse)
    
# Compute analytical and DNN's solution.
g_dnn_ag = np.zeros((Nx, Nt))
G_analytical = np.zeros((Nx, Nt))
for i,x_ in enumerate(x):
    for j, t_ in enumerate(t):
        point = np.array([x_, t_])
        g_dnn_ag[i,j] = g_trial(point,P)
    
        G_analytical[i,j] = g_analytic(point)

# Compute MSE.
MSE = mean_squared_error(G_analytical, g_dnn_ag)
    
print(f'Mean squared error = {MSE}')

# Iterations array.
iters = numpy.arange(1, num_iter + 1, 1)




# 3D Plot of the heat equation solution surfase.
X, T = np.meshgrid(x,t)
fig = plt.figure()
ax = fig.gca(projection='3d')
s = ax.plot_surface(X,T,g_dnn_ag,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$u(x,t)$')
#plt.savefig(PNG_path / f'DNN_solution_3D_1000iter.png')
#plt.savefig(PDF_path / f'DNN_solution_3D_1000iter.pdf')




# Taking slices from the analytical and DNN's solution.
indx1 = 0
indx2 = int(Nt/4)
indx3 = Nt - 1

t1 = t[indx1]
t2 = t[indx2]
t3 = t[indx3]

# Slice the results from the DNN
res1 = g_dnn_ag[:,indx1]
res2 = g_dnn_ag[:,indx2]
res3 = g_dnn_ag[:,indx3]

# Slice the analytical results
res_analytical1 = G_analytical[:,indx1]
res_analytical2 = G_analytical[:,indx2]
res_analytical3 = G_analytical[:,indx3]

# Slices plot.
plt.figure()
plt.style.use('ggplot')
plt.plot(x, res1, label = 'DNN solution, $t = 0$')
plt.plot(x,res_analytical1, label = 'analytical solution, $t = 0$')
plt.legend()
#plt.savefig(PNG_path / f'DNN_solution_slice1_100iter.png')
#plt.savefig(PDF_path / f'DNN_solution_slice1_100iter.pdf')

plt.figure()
plt.plot(x, res2, label = 'DNN solution, $t = 0.25$')
plt.plot(x,res_analytical2, label = 'analytical solution, $t = 0.25$')
plt.legend()
#plt.savefig(PNG_path / f'DNN_solution_slice2_100iter.png')
#plt.savefig(PDF_path / f'DNN_solution_slice2_100iter.pdf')

plt.figure()
plt.plot(x, res3, label = 'DNN solution, $t = 0.25$')
plt.plot(x,res_analytical3, label = 'analytical solution, $t = 1$')
plt.legend()
#plt.savefig(PNG_path / f'DNN_solution_slice3_100iter.png')
#plt.savefig(PDF_path / f'DNN_solution_slice3_100iter.pdf')

plt.show()