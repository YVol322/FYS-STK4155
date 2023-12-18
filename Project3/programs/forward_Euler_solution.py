import pathlib
import numpy as np
import matplotlib.pyplot as plt
from functions import Forward_Euler, analytical_solution
from sklearn.metrics import mean_squared_error, r2_score

current_path = pathlib.Path.cwd()

figures_path = current_path / 'figures'
PNG_path = figures_path / 'PNG'
PDF_path = figures_path / 'PDF'

# Specific combinations
Nx_values = [10, 100]
Nt_values = [1000, 100000]

for Nx, Nt in zip(Nx_values, Nt_values):

    L = 1.0
    T = 1.0
    a = 1.0
    
    dx = L / Nx
    dt = T / Nt

    x_values = np.linspace(0, L, Nx+1)
    t_values = np.linspace(0, T, Nt+1)
    u_FE = np.zeros((Nx+1, Nt+1))

    u_FE[:, 0] = np.sin(np.pi * x_values)
    u_FE[0, :] = 0
    u_FE[-1, :] = 0

    u_FE = Forward_Euler(u_FE, Nt, Nx, dt, dx)

    u_an = np.zeros((Nx+1, Nt+1))

    for i, x in enumerate(x_values):
        for j, t in enumerate(t_values):
            u_an[i, j] = analytical_solution(x, t, L, a)

    MSE = mean_squared_error(u_an, u_FE)
    R2 = r2_score(u_an, u_FE)

    print(f'For Nx={Nx}, Nt={Nt}:')
    print(f'Mean squared error = {MSE}')
    print(f'R2 score = {R2}')

    t1 = T/4
    t1_index = int(t1 * Nt)
    u_FE1 = u_FE[:, t1_index]
    u_an1 = u_an[:, t1_index]

    t2 = T
    t2_index = int(t2 * Nt)
    u_FE2 = u_FE[:, t2_index]
    u_an2 = u_an[:, t2_index]



    if Nx == 10:
        X, T = np.meshgrid(x_values, np.linspace(0, T, Nt+1))
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, T, u_FE.T, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('Time')
        ax.set_zlabel('u(x,t)')
        plt.savefig(PNG_path / 'FE_solution_3D.png')
        plt.savefig(PDF_path / 'FE_solution_3D.pdf')

    plt.figure()
    plt.style.use('ggplot')
    plt.plot(x_values, u_an1, label = 'analytical solution, $t = 0.25$')
    plt.plot(x_values, u_FE1, '--', label = 'Forward Euler solution, $t = 0.25$')
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    plt.legend()
    plt.savefig(PNG_path / f'FE_solution_slice1_dx={dx}.png')
    plt.savefig(PDF_path / f'FE_solution_slice1_dx={dx}.pdf')

    plt.figure()
    plt.plot(x_values, u_an2, label = 'analytical solution, $t = 1$')
    plt.plot(x_values, u_FE2, '--', label = 'Forward Euler solution, $t = 1$')
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    plt.legend()
    plt.savefig(PNG_path / f'FE_solution_slice2_dx={dx}.png')
    plt.savefig(PDF_path / f'FE_solution_slice2_dx={dx}.pdf')
    #plt.show()