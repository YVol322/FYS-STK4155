import numpy
import pathlib
from matplotlib import cm
import autograd.numpy as np
import autograd.numpy.random as npr
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from functions import solve_pde_deep_neural_network, g_trial, g_analytic

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
num_hidden_neurons = [250, 10]
num_iter = 1000
lmb = 0.01


P = solve_pde_deep_neural_network(x,t, num_hidden_neurons, num_iter, lmb)
    
# Store the results
g_dnn_ag = np.zeros((Nx, Nt))
G_analytical = np.zeros((Nx, Nt))
for i,x_ in enumerate(x):
    for j, t_ in enumerate(t):
        point = np.array([x_, t_])
        g_dnn_ag[i,j] = g_trial(point,P)
    
        G_analytical[i,j] = g_analytic(point)
    
MSE = mean_squared_error(G_analytical, g_dnn_ag)
    
print(f'Mean squared error = {MSE}')

iters = numpy.arange(1, num_iter + 1, 1)

    
# Find the map difference between the analytical and the computed solution
diff_ag = np.abs(g_dnn_ag - G_analytical)
print('Max absolute difference between the analytical solution and the network: %g'%np.max(diff_ag))

## Plot the solutions in two dimensions, that being in position and time

T,X = np.meshgrid(t,x)

fig = plt.figure()
ax = fig.gca(projection='3d')
s = ax.plot_surface(X,T,g_dnn_ag,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$u(x,t)$')
plt.savefig(PNG_path / f'DNN_solution_3D_100iter.png')
plt.savefig(PDF_path / f'DNN_solution_3D_100iter.pdf')



fig = plt.figure()
ax = fig.gca(projection='3d')
s = ax.plot_surface(X,T,diff_ag,linewidth=0,antialiased=False,cmap=cm.viridis)
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$u(x,t)$')
plt.savefig(PNG_path / f'DNN_solution_diff_3D_100iter.png')
plt.savefig(PDF_path / f'DNN_solution_diff_3D_100iter.pdf')

## Take some slices of the 3D plots just to see the solutions at particular times
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

# Plot the slices
plt.figure()
plt.style.use('ggplot')
plt.plot(x, res1, label = 'DNN solution, $t = 0$')
plt.plot(x,res_analytical1, label = 'analytical solution, $t = 0$')
plt.legend()
plt.savefig(PNG_path / f'DNN_solution_slice1_100iter.png')
plt.savefig(PDF_path / f'DNN_solution_slice1_100iter.pdf')

plt.figure()
plt.plot(x, res2, label = 'DNN solution, $t = 0.25$')
plt.plot(x,res_analytical2, label = 'analytical solution, $t = 0.25$')
plt.legend()
plt.savefig(PNG_path / f'DNN_solution_slice2_100iter.png')
plt.savefig(PDF_path / f'DNN_solution_slice2_100iter.pdf')

plt.figure()
plt.plot(x, res3, label = 'DNN solution, $t = 0.25$')
plt.plot(x,res_analytical3, label = 'analytical solution, $t = 1$')
plt.legend()
plt.savefig(PNG_path / f'DNN_solution_slice3_100iter.png')
plt.savefig(PDF_path / f'DNN_solution_slice3_100iter.pdf')

#plt.show()