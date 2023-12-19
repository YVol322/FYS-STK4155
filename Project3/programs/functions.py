import pathlib
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import jacobian,hessian,grad
from sklearn.metrics import mean_squared_error




# This function creates a repository for a give path.
#
# Input: string path - path to the directory, which will be created.
def Create_repo(path):

    figures_path = path
    figures_path.mkdir(exist_ok=True) # Creates a directory.




# This function computes analytical solution of the heat/diffusial equation.
#
# Input: x - array or value of the position on the rod;
#        T - array or value of the time;
#        double L - length of the rod;
#        double a - thermal diffusity of the medium.
#
# Output: array u - array with analytical solution of the heat equation.
def analytical_solution(x, t, L, a):
    return np.sin(np.pi * x / L) * np.exp(-(np.pi / L)**2 * a * t)




# This function implements the explicit Forward Euler algorithm for solving PDEs.
#
# Input: u - array, that contains initail and boundary conditions.;
#        It will contain the solution of the heat equation;
#        Nt - number of time discretization points;
#        Nx - number of x coordinate discretization points;
#        dt - time step;
#        dx - stepsize along x axis.
#
# Output: array u - array with FE solution of the heat equation.
def Forward_Euler(u, Nt, Nx, dt, dx):
    for t in range(Nt):
        for i in range(1, Nx):
            u[i, t+1] = u[i, t] + dt / dx**2 * (u[i+1, t] - 2*u[i, t] + u[i-1, t]) # FE algo.

    return u




# This is a sigmoid activation function.
#
# Input: z - input to the hidden layer.
#
# Output: f - output from the hidden layer using sigmoid activation function.
def sigmoid(z):
    return 1/(1 + np.exp(-z))




# This is a hyperbolic tangent activation function.
#
# Input: z - input to the hidden layer.
#
# Output: f - output from the hidden layer using tanh activation function.
def tanh(z):
    return np.tanh(z)




# This is a RELU activation function.
#
# Input: z - input to the hidden layer.
#
# Output: f - output from the hidden layer using RELU activation function.
def relu(z):
    return np.maximum(0, z)




# This function constructs an architecture of the DNN and performs a Feed Forward step to make a prediction.
#
# Input: array deep_params - weights and biases of the DNN;
#        x - coorinate-time point (x,t).
#
# Output: Constructs the DNN with a given number of layers and nodes (shape of deep_params);
#         Returns a prediction of the DNN.
def deep_neural_network(deep_params, x):

    # x is a point and a 1D numpy array. Making it a column vector.
    num_coordinates = np.size(x,0)
    x = x.reshape(num_coordinates,-1)

    num_points = np.size(x,1) # Number of poins in the x array.

    # N_umber of hidden layers (without output layer).
    N_hidden = np.size(deep_params) - 1

    # Iput layer just takes x without doing anything to it.
    x_input = x
    x_prev = x_input

    # Hidden layers Feed Forward stage.
    for l in range(N_hidden):

        # Collect l-th layer weights from the deep_params array.
        w_hidden = deep_params[l]

        # Include bias.
        x_prev = np.concatenate((np.ones((1,num_points)), x_prev ), axis = 0)

        # Generate input and output from the l-th layer.
        z_hidden = np.matmul(w_hidden, x_prev)
        x_hidden = sigmoid(z_hidden)

        # Comment line 121 and uncomment line 125 if you want to use tanh.
        # Uncomment line 126 if you want to use relu.
        #x_hidden = tanh(z_hidden)
        #x_hidden = relu(z_hidden)


        # Update x_prev such that next layer can use the output from this layer.
        x_prev = x_hidden

    # Output layer.

    # Collect outut layer weights from the deep_params array.
    w_output = deep_params[-1]

    # Include bias.
    x_prev = np.concatenate((np.ones((1,num_points)), x_prev), axis = 0)

    # Generate input and output from the output layer.
    z_output = np.matmul(w_output, x_prev)
    x_output = z_output

    return x_output[0][0]




# This function define a function for construction of a trial fuction.
#
# Input: x - array or value of the position on the rod.
#
# Output: u - sin(pi * x) function of the given argument x.
def u(x):
    return np.sin(np.pi*x)




# This function comutes DNN's trial solution (and final one as well, when DNN's parameters are trained).
#
# Input: point - coorinate-time point (x,t);
#        P - DNN's weights and biases.
#
# Output: u_dnn - DNN's solution of the heat equation.
def g_trial(point,P):
    x,t = point
    return (1-t)*u(x) + x*(1-x)*t*deep_neural_network(P,point)





# This functions computes RHS of the heat equation. In our case it is equal to zero.
def f(point):
    return 0.




# This function computes cost function for the heat equation.
#
# Input: P - DNN's weights and biases;
#        x - array of cooridinated on the rod;
#        t - time array.
#
# Output: cost function computed with current DNN's parameters for given x and t.
def cost_function(P, x, t):

    Nx = np.size(x) # Number of elements in x array.
    Nt = np.size(t) # Number of elements in t array.

    # Initialize cost function with 0.
    cost_sum = 0

    # function that computes jacobian matrix, constructed from the cost fucntion
    # (matrix with all possible partial derivatives) using automatic differentiation.
    g_t_jacobian_func = jacobian(g_trial)

    # function that computes hessian matrix, constructed from the cost fucntion
    # (matrix with all possible second order partial derivatives) using automatic differentiation.
    g_t_hessian_func = hessian(g_trial)

    # Loops over all elements in x and t arrays.
    for x_ in x:
        for t_ in t:

            # Define (x,t) point.
            point = np.array([x_,t_])

            # Compute trial solution in this point.
            g_t = g_trial(point,P)

            # Compute jacobian and hessian matrix.
            g_t_jacobian = g_t_jacobian_func(point,P)
            g_t_hessian = g_t_hessian_func(point,P)

            # Computes dg/dt at the given point.
            g_t_dt = g_t_jacobian[1]

            # Computes d^2g/dx^2 at the given point.
            g_t_d2x = g_t_hessian[0][0]

            # Computes RHS of the heat equation.
            func = f(point)

            # Compute prediction error.
            err_sqr = ( (g_t_dt - g_t_d2x) - func)**2

            # Add current point prediction error to cost_sum.
            cost_sum += err_sqr

    # Cost function is equal to cost_sum divided by number of points in x and t arrays.
    return cost_sum /(Nx * Nt)




# This function computes analytical solution (without L and a as aruments).
#
# Input: point - coordinate-time (x,t) point.
#
# Output: g_an - analytical solution at this point.
def g_analytic(point):

    x,t = point

    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)




# This function trains DNN to solve the heat equation.
#
# Input: x - x coordinates array, on which the heat equation will be solved;
#        t - time array, for which the heat equation will be solved;
#        num_neurons - array that contains number of hidden nodes for all hidden layers;
#        num_iter - number of iterations for training DNN;
#        lmb - learning rate of the DNN;
#        bool mse - if mse == 1, returns DNN's parameters and list with MSE for all iterations.
#                   if mse == 0, returns only DNN's parameters;
#
# Output: P - trained NNs parameters.
def solve_pde_deep_neural_network(x,t, num_neurons, num_iter, lmb, mse):

    Nx = np.size(x) # Number of elements in x array.
    Nt = np.size(t) # Number of elements in t array.

    # Number of hidden layers.
    N_hidden = np.size(num_neurons)

    # Set up initial weigths and biases

    # Initialize the list of parameters.
    P = [None]*(N_hidden + 1) # + 1 to include the output layer

    P[0] = npr.randn(num_neurons[0], 2 + 1 ) # 2 since we have two points, +1 to include bias.
    for l in range(1,N_hidden):
        P[l] = npr.randn(num_neurons[l], num_neurons[l-1] + 1) # +1 to include bias.

    # For the output layer
    P[-1] = npr.randn(1, num_neurons[-1] + 1 ) # +1 since bias is included.

    print('Initial cost: ',cost_function(P, x, t))

    # Compute gradient of the cost function wrt to weights and biases.
    cost_function_grad = grad(cost_function,0)

    # Initilize list, that will contain MSE for all iterations.
    MSEs = []

    # Train DNN for num_iter number of iterations.
    for i in range(num_iter):

        print(i)

        # Comput gradient of the cost function wrt to network parameters.
        cost_grad =  cost_function_grad(P, x , t)

        # Update DNN's paramenters using gradient descent.
        for l in range(N_hidden+1):
            P[l] = P[l] - lmb * cost_grad[l]
        

        # Initialize arrays that will contain DNN's and analytical solution.
        g_dnn_ag = np.zeros((Nx, Nt))
        G_analytical = np.zeros((Nx, Nt))

        # Fill the arays
        # Loop over all elements in x and t arrays.
        for i,x_ in enumerate(x):
            for j, t_ in enumerate(t):

                point = np.array([x_, t_])
                g_dnn_ag[i,j] = g_trial(point,P) # Analytical solition at the given point.
    
                G_analytical[i,j] = g_analytic(point) # DNN's prediction at the given point.

        # Save current iteration MSE to the MSEs list.
        MSEs.append(mean_squared_error(G_analytical, g_dnn_ag))

    print('Final cost: ',cost_function(P, x, t))


    if mse == 1:
        return P, MSEs
    
    else:
        return P
