import pathlib
import numpy as np
import matplotlib.pyplot as plt
from functions import Create_repo, analytical_solution




current_path = pathlib.Path.cwd() # Currect working directory path.

# Pathes, where the figures will be saved to.
figures_path = current_path / 'figures'
PNG_path = figures_path / 'PNG'
PDF_path = figures_path / 'PDF'

# Create these repositories, if they were not created before.
Create_repo(figures_path)
Create_repo(PDF_path)
Create_repo(PNG_path)

L = 1.0 # Rod length.
T = 1.0 # Time duration, for which the heat equation will be solved.
a = 1.0 # Thermal diffusity of the medium.
Nx = 100 # Number of point in (0, 1) x coordinate array.
Nt = 100 # Number of point in (0, 1) time array.

# Fill x and t arrays.
x_values = np.linspace(0, L, Nx+1)
t_values = np.linspace(0, T, Nt+1)

# Compute analytical solution at t = 0.
t0 = 0
u0 = np.zeros(x_values.shape)
u0 = analytical_solution(x_values, t0, L, a)

# Compute analytical solution at t = 0.25.
t1 = T/4
u1 = np.zeros(x_values.shape)
u1 = analytical_solution(x_values, t1, L, a)

# Compute analytical solution at t = 1.
t2 = T
u2 = np.zeros(x_values.shape)
u2 = analytical_solution(x_values, t2, L, a)


# Compute analytical solution using x and t arrays as a meshgrid.
X, T = np.meshgrid(x_values, t_values)
analytical_solution_values = analytical_solution(X, T, L, a)



# 3D Plot of the heat equation solution surfase.
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, analytical_solution_values, cmap='viridis')
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$u(x,t)$')
#plt.savefig(PNG_path / 'analytical_solution_3D.png')
#plt.savefig(PDF_path / 'analytical_solution_3D.pdf')




# 3D Plot of the heat equation with a = 0.1 solution surfase.
a = 0.1
X, T = np.meshgrid(x_values, t_values)
analytical_solution_values = analytical_solution(X, T, L, a)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, analytical_solution_values, cmap='viridis')
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$u(x,t)$')
#plt.savefig(PNG_path / 'analytical_solution_3D_a=0.1.png')
#plt.savefig(PDF_path / 'analytical_solution_3D_a=0.1.pdf')




# Plot of the solution surface slices at the t=0, t=0.25 and t=1 moments.
plt.figure()
plt.style.use('ggplot')
plt.plot(x_values, u0, label = '$t_0 = 0$')
plt.plot(x_values, u1, label = '$t_1 = 0.25$')
plt.plot(x_values, u2, label = '$t_2 = 1$')
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.legend()
#plt.savefig(PNG_path / 'analytical_solution_slices.png')
#plt.savefig(PDF_path / 'analytical_solution_slices.pdf')
plt.show()