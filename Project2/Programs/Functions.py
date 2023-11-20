import autograd.numpy as np
from sklearn.metrics import mean_squared_error
from autograd import grad
import pathlib
import time


# This file can not be executed. It contains decralation of all function that are used in other programs.



# This function generates data for regression problem.
#
# Output: np.array of shape (10000, ) x - x values array;
#         np.array of shape (10000, ) y - y(x) = 3 - 5x + 4x^2 values array;
#         np.array of shape (10000, 3) X - design matrix for 2nd order polynomial fit.
def Data():

    n = 10000 # Number of data point.
    features = 3 # Features of design matrix.

    x = np.arange(0, 1, 1/n).reshape(-1,1) # Initialising x values array.
    y = 3 - 5 * x + 4 * x ** 2 # Computing y values array according to rule y(x) = 3 - 5x + 4x^2.

    X = np.zeros((n, features)) # Initilizing descign matrix with zeros.

    # Filling design matrix.
    for i in range(features): 
        X[:, i] = (x**i).ravel()

    return x, y, X





# This function creates directories, if they are not created yet. Returns PNG and PDF directories pathes.
# .png figures will be saved in PNG directoies and .pdf figures will be saved in PDF directories.
#
# All programs SHOULD be runned from "Project2 / Programs" directory.
#
# Input: string dir_name - name of the directory that will be crated.
#
# Output: string PNG_path - path to PNG directory "Project2 / Figures / dir_name / PNG";
#         string PDF_path - path to PDF directory "Project2 / Figures / dir_name / PDF".
def Create_dir(dir_name):

    current_path = pathlib.Path.cwd().resolve() # Current path. 

    # Creating "Figures" directory in parent directory.
    figures_path = current_path.parent / 'Figures'
    figures_path.mkdir(exist_ok=True, parents=True)

    # Creating "Figures" directory in parent directory.
    figures_path = figures_path / dir_name
    figures_path.mkdir(exist_ok=True, parents=True)

    # Creating "PNG" directory in "Figures" directoryy
    PNG_path = figures_path / 'PNG'
    PNG_path.mkdir(exist_ok=True, parents=True)

    # Creating "PDF" directory in "Figures" directoryy
    PDF_path = figures_path / 'PDF'
    PDF_path.mkdir(exist_ok=True, parents=True)

    return PNG_path, PDF_path





# This function is used to compute decreasing learning rate.
#
# Input: double t_0 - decreasing learning rate parameter;
#        double t_1 - decreasing learning rate parameter;
#        double t - iteration, on which learning rate is computed.
#
# Output: double gamma - decreasing learning rate for current iteration t.
def learning_schedule(t, t0, t1):

    gamma = t0/(t+t1) # definition of decreasing learning rate.

    return gamma





# This function computes Ridge Regression or Linear Regression (lmb = 0) cost fucntion.
#
# Input: np.array of shape (3, 1) - predicted fit coefficients;
#        double lmb - penalty parameter for Ridge rRegression (or Linear if lmb = 0);
#        int n - number of data points;
#        np.array y - target / array of y(x) values;
#        np.array X - design matrix for 2nd order fit.
#
# Output: np.array of shape (10000, ) C - cost function.
def CostRidge(beta, lmb, n, y, X):

    C = (1.0/n) * np.sum((y - X @ beta)**2) + lmb * np.sum(beta**2) # definition of cost function.

    return C





# This function implements Gradient Descent algorithm until MSE is smaller then stopping criterion.
#
# Input: np.array X - design matrix for 2nd order fit.
#        np.array y - target / array of y(x) values;
#        int f - number of features of design matrix = 3;
#        int n - number of data points;
#        double eps - stopping criterion;
#        double eta - constant learning rate;
#        double lmb - penalty parameter;
#        bool auto - 1 to compute gradients using automatic differentiation, 
#                    0 to compute gradients using analytical expression.
#
# Output: np.array of shape (3, 1) beta - predicted fit coefficients;
#         int i - number of iterations until convergence;
#         double elapsed_time - runtime of algorith in s.
def GD(X, y, f, n, eps, eta, lmb, auto):

    I = np.eye(f) # identity matrix of size (f, f) for computing beta_Ridge
    beta_Ridge = np.linalg.inv(X.T @ X + lmb * I) @ X.T @ y # Ridge Regression optimal coefs prediction (target)
    beta = np.random.randn(f,1) # initial guess of fit coefs.

    training_gradient = grad(CostRidge, 0) # Function that computes gradients using auto diff.

    i=0 # Initilising iteration number with zero.

    start_time = time.time() # Starting coutdown.

    # Running loop until MSE is smaller then eps
    while(mean_squared_error(beta_Ridge, beta)>eps):

        # Computing gradients uing auto diff if auto == 1.
        if(auto == 1): gradient = training_gradient(beta,lmb, n, y, X)

        # Computing gradients using analyt expression if auto != 1.
        else: gradient = 2/n * X.T @ (X @ beta - y) + 2 * lmb * I @ beta

        beta -= eta*gradient # GD update.

        i+=1 # Adding one to iteration number.

    elapsed_time = time.time() - start_time # Finishing coutdown.
    
    return beta, i, elapsed_time





# This function implements AdaGrad Gradient Descent algorithm until MSE is smaller then stopping criterion.
#
# Input: np.array X - design matrix for 2nd order fit.
#        np.array y - target / array of y(x) values;
#        int f - number of features of design matrix = 3;
#        int n - number of data points;
#        double eps - stopping criterion;
#        double eta - constant learning rate;
#        double lmb - penalty parameter;
#
# Output: np.array of shape (3, 1) beta - predicted fit coefficients;
#         int i - number of iterations until convergence.
def Ada(X, y, f, n, eps, eta, lmb):

    I = np.eye(f) # identity matrix of size (f, f) for computing beta_Ridge
    beta_Ridge = np.linalg.inv(X.T @ X + lmb * I) @ X.T @ y # Ridge Regression optimal coefs prediction (target)
    beta = np.random.randn(f,1) # initial guess of fit coefs.
    
    G = np.diag(np.zeros(f)) # Intilizing G matrix of shape (f,f) with zeros.
    delta = 1e-7 # Small parameter to avoid division by 0 error.

    i=0 # Initilising iteration number with zero.

    # Running loop until MSE is smaller then eps
    while(mean_squared_error(beta_Ridge, beta)>eps):

        # Computing gradients uing analyt expression (it is faster then auto diff).
        gradient = 2/n * X.T @ (X @ beta - y) + lmb * I @ beta

        # AdaGrad rule to compute learning rate.
        G += gradient*gradient
        G_diag = G.diagonal()
        sqrt_G = np.sqrt(G_diag)
        gamma = eta/(delta + sqrt_G).reshape(-1,1)
        
        beta -= gamma*gradient # GD update.

        i+=1 # Adding one to iteration number.

    return beta, i





# This function implements RMSProp Gradient Descent algorithm until MSE is smaller then stopping criterion.
#
# Input: np.array X - design matrix for 2nd order fit.
#        np.array y - target / array of y(x) values;
#        int f - number of features of design matrix = 3;
#        int n - number of data points;
#        double eps - stopping criterion;
#        double eta - constant learning rate;
#        double rho - RMSProp parameter = 0.999 or 0.99 (not smaller);
#        double lmb - penalty parameter;
#
# Output: np.array of shape (3, 1) beta - predicted fit coefficients;
#         int i - number of iterations until convergence.
def RMS(X, y, f, n, eps, eta, rho, lmb):

    I = np.eye(f) # identity matrix of size (f, f) for computing beta_Ridge
    beta_Ridge = np.linalg.inv(X.T @ X + lmb * I) @ X.T @ y # Ridge Regression optimal coefs prediction (target)
    beta = np.random.randn(f,1) # initial guess of fit coefs.
    
    G = np.diag(np.zeros(f)) # Intilizing G matrix of shape (f,f) with zeros.
    delta = 1e-7 # Small parameter to avoid division by 0 error.

    i=0 # Initilising iteration number with zero.

    # Running loop until MSE is smaller then eps
    while(mean_squared_error(beta_Ridge, beta)>eps):

        # Computing gradients uing analyt expression (it is faster then auto diff).
        gradient = 2/n * X.T @ (X @ beta - y) + lmb * I @ beta

        # RMSProp rule to compute learning rate.
        G = (rho*G+(1-rho)*gradient*gradient)
        G_diag = G.diagonal()
        sqrt_G = np.sqrt(G_diag)
        gamma = eta/(delta + sqrt_G).reshape(-1,1)

        beta -= gamma*gradient # GD update.

        i+=1 # Adding one to iteration number.

    return beta, i





# This function implements ADAM Gradient Descent algorithm until MSE is smaller then stopping criterion.
#
# Input: np.array X - design matrix for 2nd order fit.
#        np.array y - target / array of y(x) values;
#        int f - number of features of design matrix = 3;
#        int n - number of data points;
#        double eps - stopping criterion;
#        double eta - constant learning rate;
#        double beta1 - ADAM parameter = 0.99 (not smaller);
#        double beta2 - ADAM parameter = 0.999 (not smaller);
#        double lmb - penalty parameter;
#
# Output: np.array of shape (3, 1) beta - predicted fit coefficients;
#         int i - number of iterations until convergence.
def ADAM(X, y, f, n, eps, eta, beta1, beta2, lmb):

    I = np.eye(f) # identity matrix of size (f, f) for computing beta_Ridge
    beta_Ridge = np.linalg.inv(X.T @ X + lmb * I) @ X.T @ y # Ridge Regression optimal coefs prediction (target)
    beta = np.random.randn(f,1) # initial guess of fit coefs.

    # Intializing ADAM variables used for update with zeros.
    first_moment = 0.0
    second_moment = 0.0
    delta = 1e-7 # Small parameter to avoid division by 0 error.

    i=0 # Initilising iteration number with zero.

    # Running loop until MSE is smaller then eps
    while(mean_squared_error(beta_Ridge, beta)>eps):

        # Computing gradients uing analyt expression (it is faster then auto diff).
        gradient = 2/n * X.T @ (X @ beta - y) + lmb * I @ beta

        # ADAM updating rule.
        first_moment = beta1*first_moment + (1-beta1)*gradient
        second_moment = beta2*second_moment+(1-beta2)*gradient*gradient

        first_term = first_moment/(1.0-beta1**(i+1))
        second_term = second_moment/(1.0-beta2**(i+1))

        update = eta*first_term/(np.sqrt(second_term)+delta)

        beta -= update # GD update (it is alredy multiplied with eta already).
        
        i+=1 # Adding one to iteration number.

    return beta, i





# This function implements Gradient Descent with momentum algorithm until
# MSE is smaller then stopping criterion.
#
# Input: np.array X - design matrix for 2nd order fit.
#        np.array y - target / array of y(x) values;
#        int f - number of features of design matrix = 3;
#        int n - number of data points;
#        double eps - stopping criterion;
#        double eta - constant learning rate;
#        double moment - momentum;
#        double lmb - penalty parameter;
#
# Output: np.array of shape (3, 1) beta - predicted fit coefficients;
#         int i - number of iterations until convergence.
def GD_momentum(X, y, f, n, eps, eta, moment, lmb):

    I = np.eye(f) # identity matrix of size (f, f) for computing beta_Ridge
    beta_Ridge = np.linalg.inv(X.T @ X + lmb * I) @ X.T @ y # Ridge Regression optimal coefs prediction (target)
    beta = np.random.randn(f,1) # initial guess of fit coefs.

    change = 0 # Initializing previous iterations update with zero.

    i=0 # Initilising iteration number with zero.

    # Running loop until MSE is smaller then eps
    while(mean_squared_error(beta_Ridge, beta)>eps):

        # Computing gradients uing analyt expression (it is faster then auto diff).
        gradient = 2/n * X.T @ (X @ beta - y) + lmb * I @ beta

        # GD with momentum update.
        new_change = eta*gradient + moment * change

        beta -= new_change
        change = new_change

        i+=1 # Adding one to iteration number.
    
    return beta, i





# This function implements AdaGrad Gradient Descent with momentum algorithm until
# MSE is smaller then stopping criterion.
#
# Input: np.array X - design matrix for 2nd order fit.
#        np.array y - target / array of y(x) values;
#        int f - number of features of design matrix = 3;
#        int n - number of data points;
#        double eps - stopping criterion;
#        double eta - constant learning rate;
#        double moment - momentum;
#
# Output: np.array of shape (3, 1) beta - predicted fit coefficients;
#         int i - number of iterations until convergence.
def Ada_momentum(X, y, f, n, eps, eta, moment):

    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y # Linear Regression optimal coefs prediction (target)
    beta = np.random.randn(f,1) # initial guess of fit coefs.

    G = np.diag(np.zeros(f)) # Intilizing G matrix of shape (f,f) with zeros.
    delta = 1e-7 # Small parameter to avoid division by 0 error.

    change = 0 # Initializing previous iterations update with zero.

    i=0 # Initilising iteration number with zero.

    # Running loop until MSE is smaller then eps
    while(mean_squared_error(beta_linreg, beta)>eps):

        # Computing gradients uing analyt expression (it is faster then auto diff).
        gradient = (2.0/n)*X.T @ (X @ beta-y)

        # AdaGrad with momentum update.
        G += gradient*gradient
        G_diag = G.diagonal()
        sqrt_G = np.sqrt(G_diag)

        gamma = eta/(delta + sqrt_G).reshape(-1,1)

        new_change = gamma*gradient + moment * change

        beta -= new_change
        change = new_change

        i+=1 # Adding one to iteration number.

    return beta, i





# This function implements RMSProp Gradient Descent with momentum algorithm until
# MSE is smaller then stopping criterion.
#
# Input: np.array X - design matrix for 2nd order fit.
#        np.array y - target / array of y(x) values;
#        int f - number of features of design matrix = 3;
#        int n - number of data points;
#        double eps - stopping criterion;
#        double eta - constant learning rate;
#        double rho - RMSProp parameter = 0.999 or 0.99 (not smaller);
#        double moment - momentum;
#
# Output: np.array of shape (3, 1) beta - predicted fit coefficients;
#         int i - number of iterations until convergence.
def RMS_momentum(X, y, f, n, eps, eta, rho, moment):
    
    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y # Linear Regression optimal coefs prediction (target)
    beta = np.random.randn(f,1) # initial guess of fit coefs.

    G = np.diag(np.zeros(f)) # Intilizing G matrix of shape (f,f) with zeros.
    delta = 1e-7 # Small parameter to avoid division by 0 error.

    change = 0 # Initializing previous iterations update with zero.

    i=0 # Initilising iteration number with zero.

    # Running loop until MSE is smaller then eps
    while(mean_squared_error(beta_linreg, beta)>eps):

        # Computing gradients uing analyt expression (it is faster then auto diff).
        gradient = (2.0/n)*X.T @ (X @ beta-y)

        # RMSProp with momentum update.
        G = (rho*G+(1-rho)*gradient*gradient)
        G_diag = G.diagonal()
        sqrt_G = np.sqrt(G_diag)
        gamma = eta/(delta + sqrt_G).reshape(-1,1)

        new_change = gamma*gradient + moment * change

        beta -= new_change
        change = new_change

        i+=1 # Adding one to iteration number.

    return beta, i





# This function implements ADAM Gradient Descent with momentum algorithm until
# MSE is smaller then stopping criterion.
#
# Input: np.array X - design matrix for 2nd order fit.
#        np.array y - target / array of y(x) values;
#        int f - number of features of design matrix = 3;
#        int n - number of data points;
#        double eps - stopping criterion;
#        double eta - constant learning rate;
#        double beta1 - ADAM parameter = 0.99 (not smaller);
#        double beta2 - ADAM parameter = 0.999 (not smaller);
#        double moment - momentum;
#
# Output: np.array of shape (3, 1) beta - predicted fit coefficients;
#         int i - number of iterations until convergence.
def ADAM_momentum(X, y, f, n, eps, eta, beta1, beta2, moment):

    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y # Linear Regression optimal coefs prediction (target)
    beta = np.random.randn(f,1) # initial guess of fit coefs.

    # Intializing ADAM variables used for update with zeros.
    first_moment = 0.0
    second_moment = 0.0

    delta = 1e-7 # Small parameter to avoid division by 0 error.

    change = 0 # Initializing previous iterations update with zero.
    
    i=0 # Initilising iteration number with zero.

    # Running loop until MSE is smaller then eps
    while(mean_squared_error(beta_linreg, beta)>eps):
        
        # Computing gradients uing analyt expression (it is faster then auto diff).
        gradient = (2.0/n)*X.T @ (X @ beta-y)

        # ADAM with momentum update.
        first_moment = beta1*first_moment + (1-beta1)*gradient
        second_moment = beta2*second_moment+(1-beta2)*gradient*gradient
        first_term = first_moment/(1.0-beta1**(i+1))
        second_term = second_moment/(1.0-beta2**(i+1))

        update = eta*first_term/(np.sqrt(second_term)+delta)

        new_change = update + moment * change

        beta -= new_change
        change = new_change
        
        i+=1 # Adding one to iteration number.

    return beta, i





# This function implements Stochastic Gradient Descent algorithm n_epoch iterations.
#
# Input: np.array X - design matrix for 2nd order fit.
#        np.array y - target / array of y(x) values;
#        int f - number of features of design matrix = 3;
#        int n - number of data points;
#        double t_0 - decreasing learning rate parameter;
#        double t_1 - decreasing learning rate parameter;
#        int M - mini-batch size;
#        int n_epoch - number of epochs;
#        np.array of shape (3, 1) beta - predicted fit coefficients.
#
# Output: np.array of shape (3, 1) beta - predicted fit coefficients.
def SGD(X, y, f, n, t0, t1, M, n_epoch, beta):

    m = int(n/M) # number of mini-batches;

    # Loop over epochs.
    for epoch in range(n_epoch):

        # Loop over mini-bathces
        for i in range(m):

            r = M * np.random.randint(m) # choosing random integer.

            # Making smaller arrays of initial ones.
            Xi = X[r : r + M]
            yi = y[r : r + M]

            # Computing gradients uing analyt expression (it is faster then auto diff).
            gradients = (2.0/M)* Xi.T @ ((Xi @ beta)-yi)

            # Computing decreasing leartning rate at epoch*m+i iteration.
            eta = learning_schedule(epoch*m+i, t0, t1)

            beta -= eta*gradients # GD update.
    
    return beta





# This function implements AdaGrad Stochastic Gradient Descent algorithm n_epoch iterations.
#
# Input: np.array X - design matrix for 2nd order fit.
#        np.array y - target / array of y(x) values;
#        int f - number of features of design matrix = 3;
#        int n - number of data points;
#        double t_0 - decreasing learning rate parameter;
#        double t_1 - decreasing learning rate parameter;
#        int M - mini-batch size;
#        int n_epoch - number of epochs;
#        np.array of shape (3, 1) beta - predicted fit coefficients.
#
# Output: np.array of shape (3, 1) beta - predicted fit coefficients.
def SGD_Ada(X, y, f, n, t0, t1, M, n_epoch, beta):

    G = np.diag(np.zeros(f)) # Intilizing G matrix of shape (f,f) with zeros.
    delta = 1e-7 # Small parameter to avoid division by 0 error.

    m = int(n/M) # number of mini-batches;

    # Loop over epochs.
    for epoch in range(n_epoch):

        # Loop over mini-bathces
        for i in range(m):

            r = M * np.random.randint(m) # choosing random integer.

            # Making smaller arrays of initial ones.
            Xi = X[r : r + M]
            yi = y[r : r + M]

            # Computing gradients uing analyt expression (it is faster then auto diff).
            gradients = (2.0/M)* Xi.T @ ((Xi @ beta)-yi)

            # AdaGrad rule to compute learning rate.
            G += gradients*gradients
            G_diag = G.diagonal()
            sqrt_G = np.sqrt(G_diag)

            # Computing decreasing leartning rate at epoch*m+i iteration.
            eta = learning_schedule(epoch*m+i, t0, t1)
            gamma = eta/(delta + sqrt_G).reshape(-1,1)

            beta -= gamma*gradients # GD update.

    return beta





# This function implements RMSProp Stochastic Gradient Descent algorithm n_epoch iterations.
#
# Input: np.array X - design matrix for 2nd order fit.
#        np.array y - target / array of y(x) values;
#        int f - number of features of design matrix = 3;
#        int n - number of data points;
#        double rho - RMSProp parameter = 0.999 or 0.99 (not smaller);
#        double t_0 - decreasing learning rate parameter;
#        double t_1 - decreasing learning rate parameter;
#        int M - mini-batch size;
#        int n_epoch - number of epochs;
#        np.array of shape (3, 1) beta - predicted fit coefficients.
#
# Output: np.array of shape (3, 1) beta - predicted fit coefficients.
def SGD_RMS(X, y, f, n, rho, t0, t1, M, n_epoch, beta):

    G = np.diag(np.zeros(f)) # Intilizing G matrix of shape (f,f) with zeros.
    delta = 1e-7 # Small parameter to avoid division by 0 error.

    m = int(n/M) # number of mini-batches;

    # Loop over epochs.
    for epoch in range(n_epoch):

        # Loop over mini-bathces
        for i in range(m):

            r = M * np.random.randint(m) # choosing random integer.

            # Making smaller arrays of initial ones.
            Xi = X[r : r + M]
            yi = y[r : r + M]

            # Computing gradients uing analyt expression (it is faster then auto diff).
            gradients = (2.0/M)* Xi.T @ ((Xi @ beta)-yi)

            # RMSProp rule to compute learning rate.
            G = (rho*G+(1-rho)*gradients*gradients)
            G_diag = G.diagonal()
            sqrt_G = np.sqrt(G_diag)
            
            # Computing decreasing leartning rate at epoch*m+i iteration.
            eta = learning_schedule(epoch*m+i, t0, t1)
            gamma = eta/(delta + sqrt_G).reshape(-1,1)

            beta -= gamma*gradients # GD update.

    return beta





# This function implements ADAM Stochastic Gradient Descent algorithm n_epoch iterations.
#
# Input: np.array X - design matrix for 2nd order fit.
#        np.array y - target / array of y(x) values;
#        int f - number of features of design matrix = 3;
#        int n - number of data points;
#        double beta1 - ADAM parameter = 0.99 (not smaller);
#        double beta2 - ADAM parameter = 0.999 (not smaller);
#        double t_0 - decreasing learning rate parameter;
#        double t_1 - decreasing learning rate parameter;
#        int M - mini-batch size;
#        int n_epoch - number of epochs;
#        np.array of shape (3, 1) beta - predicted fit coefficients.
#
# Output: np.array of shape (3, 1) beta - predicted fit coefficients.
def SGD_ADAM(X, y, f, n, beta1, beta2, t0, t1, M, n_epoch, beta):

    # Intializing ADAM variables used for update with zeros.
    first_moment = 0.0
    second_moment = 0.0

    delta = 1e-7 # Small parameter to avoid division by 0 error.

    m = int(n/M) # number of mini-batches;
    
    # Loop over epochs.
    for epoch in range(n_epoch):

        # Loop over mini-bathces
        for i in range(m):

            r = M * np.random.randint(m) # choosing random integer.

            # Making smaller arrays of initial ones.
            Xi = X[r : r + M]
            yi = y[r : r + M]

            # Computing gradients uing analyt expression (it is faster then auto diff).
            gradients = (2.0/M)* Xi.T @ ((Xi @ beta)-yi)

            # ADAM rule to compute learning rate.
            first_moment = beta1*first_moment + (1-beta1)*gradients
            second_moment = beta2*second_moment+(1-beta2)*gradients*gradients
            first_term = first_moment/(1.0-beta1**(i+1))
            second_term = second_moment/(1.0-beta2**(i+1))

            # Computing decreasing leartning rate at epoch*m+i iteration.
            eta = learning_schedule(epoch*m+i, t0, t1)
            update = eta*first_term/(np.sqrt(second_term)+delta)

            beta -= update # GD update.

    return beta





# This function implements Stochastic Gradient Descent with momentum algorithm n_epoch iterations.
#
# Input: np.array X - design matrix for 2nd order fit.
#        np.array y - target / array of y(x) values;
#        int f - number of features of design matrix = 3;
#        int n - number of data points;
#        double moment - momentum;
#        double t_0 - decreasing learning rate parameter;
#        double t_1 - decreasing learning rate parameter;
#        int M - mini-batch size;
#        int n_epoch - number of epochs;
#        np.array of shape (3, 1) beta - predicted fit coefficients.
#
# Output: np.array of shape (3, 1) beta - predicted fit coefficients.
def SGDM(X, y, f, n, moment, t0, t1, M, n_epoch, beta):

    change = 0 # Initializing previous iterations update with zero.

    m = int(n/M) # number of mini-batches;

    # Loop over epochs.
    for epoch in range(n_epoch):

        # Loop over mini-bathces
        for i in range(m):

            r = M * np.random.randint(m) # choosing random integer.

            # Making smaller arrays of initial ones.
            Xi = X[r : r + M]
            yi = y[r : r + M]

            # Computing gradients uing analyt expression (it is faster then auto diff).
            gradients = (2.0/M)* Xi.T @ ((Xi @ beta)-yi)

            # Computing decreasing leartning rate at epoch*m+i iteration.
            eta = learning_schedule(epoch*m+i, t0, t1)

            # GD with momentum upate rule.
            new_change = eta*gradients + moment * change

            beta -= new_change
            change = new_change
    
    return beta





# This function implements AdaGrad Stochastic Gradient Descent with momentum algorithm n_epoch iterations.
#
# Input: np.array X - design matrix for 2nd order fit.
#        np.array y - target / array of y(x) values;
#        int f - number of features of design matrix = 3;
#        int n - number of data points;
#        double moment - momentum;
#        double t_0 - decreasing learning rate parameter;
#        double t_1 - decreasing learning rate parameter;
#        int M - mini-batch size;
#        int n_epoch - number of epochs;
#        np.array of shape (3, 1) beta - predicted fit coefficients.
#
# Output: np.array of shape (3, 1) beta - predicted fit coefficients.
def SGDM_Ada(X, y, f, n, moment, t0, t1, M, n_epoch, beta):

    G = np.diag(np.zeros(f)) # Intilizing G matrix of shape (f,f) with zeros.
    delta = 1e-7 # Small parameter to avoid division by 0 error.

    change = 0 # Initializing previous iterations update with zero.

    m = int(n/M) # number of mini-batches;

    # Loop over epochs.
    for epoch in range(n_epoch):

        # Loop over mini-bathces
        for i in range(m):

            r = M * np.random.randint(m) # choosing random integer.

            # Making smaller arrays of initial ones.
            Xi = X[r : r + M]
            yi = y[r : r + M]

            # Computing gradients uing analyt expression (it is faster then auto diff).
            gradients = (2.0/M)* Xi.T @ ((Xi @ beta)-yi)

            # AdaGrad update rule.
            G += gradients*gradients
            G_diag = G.diagonal()
            sqrt_G = np.sqrt(G_diag)

            # Computing decreasing leartning rate at epoch*m+i iteration.
            eta = learning_schedule(epoch*m+i, t0, t1)

            # AdaGrad GD with momentum upate rule.
            gamma = eta/(delta + sqrt_G).reshape(-1,1)
            new_change = gamma*gradients + moment * change

            beta -= new_change
            change = new_change

    return beta





# This function implements RMSProp Stochastic Gradient Descent with momentum algorithm n_epoch iterations.
#
# Input: np.array X - design matrix for 2nd order fit.
#        np.array y - target / array of y(x) values;
#        int f - number of features of design matrix = 3;
#        int n - number of data points;
#        double rho - RMSProp parameter = 0.999 or 0.99 (not smaller);
#        double moment - momentum;
#        double t_0 - decreasing learning rate parameter;
#        double t_1 - decreasing learning rate parameter;
#        int M - mini-batch size;
#        int n_epoch - number of epochs;
#        np.array of shape (3, 1) beta - predicted fit coefficients.
#
# Output: np.array of shape (3, 1) beta - predicted fit coefficients.
def SGDM_RMS(X, y, f, n, rho, moment, t0, t1, M, n_epoch, beta):

    G = np.diag(np.zeros(f)) # Intilizing G matrix of shape (f,f) with zeros.
    delta = 1e-7 # Small parameter to avoid division by 0 error.

    change = 0 # Initializing previous iterations update with zero.

    m = int(n/M) # number of mini-batches;

    # Loop over epochs.
    for epoch in range(n_epoch):

        # Loop over mini-bathces
        for i in range(m):

            r = M * np.random.randint(m) # choosing random integer.

            # Making smaller arrays of initial ones.
            Xi = X[r : r + M]
            yi = y[r : r + M]

            # Computing gradients uing analyt expression (it is faster then auto diff).
            gradients = (2.0/M)* Xi.T @ ((Xi @ beta)-yi)

            # RMSProp update rule
            G = (rho*G+(1-rho)*gradients*gradients)
            G_diag = G.diagonal()
            sqrt_G = np.sqrt(G_diag)
            
            # Computing decreasing leartning rate at epoch*m+i iteration.
            eta = learning_schedule(epoch*m+i, t0, t1)

            # RMSProp GD with momentum upate rule.
            gamma = eta/(delta + sqrt_G).reshape(-1,1)
            new_change = gamma*gradients + moment * change

            beta -= new_change
            change = new_change

    return beta





# This function implements ADAM Stochastic Gradient Descent with momentum algorithm n_epoch iterations.
#
# Input: np.array X - design matrix for 2nd order fit.
#        np.array y - target / array of y(x) values;
#        int f - number of features of design matrix = 3;
#        int n - number of data points;
#        double beta1 - ADAM parameter = 0.99 (not smaller);
#        double beta2 - ADAM parameter = 0.999 (not smaller);
#        double moment - momentum;
#        double t_0 - decreasing learning rate parameter;
#        double t_1 - decreasing learning rate parameter;
#        int M - mini-batch size;
#        int n_epoch - number of epochs;
#        np.array of shape (3, 1) beta - predicted fit coefficients.
#
# Output: np.array of shape (3, 1) beta - predicted fit coefficients.
def SGDM_ADAM(X, y, f, n, beta1, beta2, moment, t0, t1, M, n_epoch, beta):

    # Intializing ADAM variables used for update with zeros.
    first_moment = 0.0
    second_moment = 0.0

    delta = 1e-7 # Small parameter to avoid division by 0 error.
    
    change = 0 # Initializing previous iterations update with zero.

    m = int(n/M) # number of mini-batches;

    # Loop over epochs.
    for epoch in range(n_epoch):

        # Loop over mini-bathces
        for i in range(m):

            r = M * np.random.randint(m) # choosing random integer.

            # Making smaller arrays of initial ones.
            Xi = X[r : r + M]
            yi = y[r : r + M]

            # Computing gradients uing analyt expression (it is faster then auto diff).
            gradients = (2.0/M)* Xi.T @ ((Xi @ beta)-yi)

            # ADAM GD update rule.
            first_moment = beta1*first_moment + (1-beta1)*gradients
            second_moment = beta2*second_moment+(1-beta2)*gradients*gradients
            first_term = first_moment/(1.0-beta1**(i+1))
            second_term = second_moment/(1.0-beta2**(i+1))

            # Computing decreasing leartning rate at epoch*m+i iteration.
            eta = learning_schedule(epoch*m+i, t0, t1)

            # ADAM GD update rule.
            update = eta*first_term/(np.sqrt(second_term)+delta)
            new_change = update + moment * change

            beta -= new_change
            change = new_change

    return beta





# This function computes sigmoid function of given array.
#
# Input: np.array of shape (n, f) x - array containig input to the hidden layer.
#
# Output: np.array of shape (n, f) f - sigmoid function of this array
def sigmoid(x):

    sigm = 1/(1 + np.exp(-x)) # Sigmoid definition.

    return sigm





# This function computes derivative of sigmoid function of given array.
#
# Input: np.array x - array of delta values.
#
# Output: np.array sigm_prime - sigmoid function of this array
def sigmoid_derivative(x):

    sigm_prime = x * (1 - x) # Derivative of sigmoid definition.

    return sigm_prime





# This function computes RELU function of given array.
#
# Input: np.array of shape (n, f) x - array containig input to the hidden layer.
#
# Output: np.array of shape (n, f) R - sigmoid function of this array
def RELU(x):

    R = np.maximum(x, 0) # RELU definition

    return R 





# This function computes derivative of RELU function of given array.
#
# Input: np.array x - array of delta values.
#
# Output: np.array signm_prime - sigmoid function of this array
def RELU_derivative(x):

    R_prime = np.where(x >= 0, 1, 0) # Derivative of RELU definition.

    return R_prime





# This function computes RELU function of given array.
#
# Input: np.array of shape (n, f) x - array containig input to the hidden layer.
#
# Output: np.array of shape (n, f) L - sigmoid function of this array
def leaky_RELU(x, alpha=0.01):

    L = np.where(x >= 0, x, alpha * x) # Leaky RELU definition.

    return L





# This function computes derivative of leaky RELU function of given array.
#
# Input: np.array x - array of delta values.
#
# Output: np.array signm_prime - sigmoid function of this array
def leaky_RELU_derivative(x, alpha=0.01):

    L_prime = np.where(x >= 0, 1, alpha) # Derivative of leaky RELU definition.

    return L_prime





# Whis function computes derivative of MSE cost function.
#
# Input: np.array y_true - target values array;
#        np.array y_pred - prediction values array;
#
# Output: gradient of cost function wrt y_pred.
def Costfunction_grad(y_true, y_pred):
    return (y_pred - y_true)





# This function initilizes list of weights and biases. Weights are initialied uniformly, 
# biases are set to 0.01 values.
#
# Input: int n_features - number of design matrix features;
#        int n_hidden_nodes - number of hidden nodes of NN;
#        int n_hidden_layers - number of hidden layers of NN;
#        int n_output_nodes - number of output nodes of NN.
#
# Output: list weigths - list, that contains weidths matrixes for every layer;
#         list biases - list, that contains biases arrays for every layer.
def initialize_W_and_b(n_features, n_hidden_nodes, n_hidden_layers, n_output_nodes):
    
    # Initilize empty lists.
    weigths = []
    biases = []

    # Initilize first layer weights and biases.
    hidden_weights_1 = np.random.randn(n_features, n_hidden_nodes)
    hidden_bias_1 = np.zeros(n_hidden_nodes) + 0.01

    # Append them to the list.
    weigths.append(hidden_weights_1)
    biases.append(hidden_bias_1)

    for i in range(n_hidden_layers - 1):

        # Initilize i-th layer weights and biases.
        hidden_weights_ = np.random.randn(n_hidden_nodes, n_hidden_nodes)
        hidden_bias_ = np.zeros(n_hidden_nodes) + 0.01

        # Append them to the list.
        weigths.append(hidden_weights_)
        biases.append(hidden_bias_)

    # Initilize output layer weights and biases.
    output_weights = np.random.randn(n_hidden_nodes, n_output_nodes)
    output_bias = np.zeros(n_output_nodes) + 0.01

    # Append them to the list.
    weigths.append(output_weights)
    biases.append(output_bias)

    return weigths, biases





# This function perform Feed Forward step in order to make a regression problem prediction.
#
# Input: np.array X - x values array (or design matrix);
#        list W_list - weights list;
#        list b_list - biases list;
#        string activation - "s" for sigmoid, "r" for RELU, "l" for leaky RELU.
#
# Output: list z_list - list of inputs to every layer;
#         list a_list - list of ouputs of every layer, acept last;
def FeedForward(X, W_list, b_list, activation):

    # Initilize empty lists.
    z_list = []
    a_list = []

    # Compute first layer input and append it to the list.
    z_1 = X @ W_list[0] + b_list[0]
    z_list.append(z_1)

    # Compute first layer output and append it to the list for the chosen act function.
    if(activation == 's'): a_1 = sigmoid(z_1)
    elif(activation == 'r'): a_1 = RELU(z_1)
    elif(activation == 'l'): a_1 = leaky_RELU(z_1)
    a_list.append(a_1)

    # Loop over all layers after first.
    for i in range(len(W_list) - 1):

        # Compute i-th layer input and append it to the list.
        z_i = a_list[i] @ W_list[i+1] + b_list[i+1]
        z_list.append(z_i)

        # don't compute and append output of the output layer.
        if i == len(W_list) - 2: break

        # Compute i-th layer output and append it to the list for the chosen act function.
        if(activation == 's'): a_i = sigmoid(z_i)
        elif(activation == 'r'): a_i = RELU(z_i)
        elif(activation == 'l'): a_i = leaky_RELU(z_i)
        a_list.append(a_i)
    
    return z_list, a_list





# This function perform Back Propagation step in order to train weights and biases.
#
# Input: np.array y_train - train target array;
#        np.array X_train - train x values array (or design matrix);
#        list W_list - weights list;
#        list b_list - biases list;
#        int M - mini-batch size;
#        int n_epoch - number of epochs;
#        double t_0 - decreasing learning rate parameter;
#        double t_1 - decreasing learning rate parameter;
#        string activation - "s" for sigmoid, "r" for RELU, "l" for leaky RELU.
#
# Output: list W_list - updated weights list;
#         list B_list - updated biases list.
def StochasticBackPropagation(y_train, X_train, W_list, b_list, M, n_epoch, t0, t1, activation):

    n = X_train.shape[0] # Number of data points.

    m = int(n/M) # number of mini-batches.

    # Loop over epochs.
    for epoch in range(n_epoch):

        # Loop over mini-bathces
        for j in range(m):

            r = M * np.random.randint(m) # choosing random integer.

            # Making smaller arrays of initial ones.
            Xi = X_train[r : r + M]
            yi = y_train[r : r + M]

            # Make a FF step to get z_list and a_list for current min-batch.
            z_list, a_list = FeedForward(Xi, W_list, b_list, activation)

            # Initialize list of deltas.
            delta_list = []

            # Compute output layer delta and append it to the list.
            delta_out = Costfunction_grad(yi, z_list[-1])
            delta_list.append(delta_out)

            # Loop over all layer.
            for i in range(len(W_list) - 1):

                # Compute i-th layer delta and append it to the list using chosen ac function.
                if activation == 's':
                    delta_i = (delta_list[-1] @ (W_list[-1 - i]).T) * sigmoid_derivative(a_list[-1 - i])

                elif activation == 'r':
                    delta_i = (delta_list[-1] @ (W_list[-1 - i]).T) * RELU_derivative(a_list[-1 - i])

                elif activation == 'l':
                    delta_i = (delta_list[-1] @ (W_list[-1 - i]).T) * leaky_RELU_derivative(a_list[-1 - i])
                delta_list.append(delta_i)


            delta_list.reverse() # Reverse deltas list.

            X_train_point_T = Xi.T  # Transpose X_train points.

            gamma = learning_schedule(epoch * m + j, t0, t1)  # Compute decreasing lern rate at cur iter.

            # Update first layer weights and biases.
            W_list[0] -= gamma * (X_train_point_T @ delta_list[0])
            b_list[0] -= gamma * np.sum(delta_list[0])

            # Update i-th layer weights and biases.
            for i in range(len(W_list) - 1):
                a_list_i_T = a_list[i].T
                W_list[i + 1] -= gamma * (a_list_i_T @ delta_list[i + 1])
                b_list[i + 1] -= gamma * np.sum(delta_list[i + 1])

    return W_list, b_list





# This function perform Feed Forward step in order to make a classification problem prediction.
#
# Input: np.array X - x values array (or design matrix);
#        list W_list - weights list;
#        list b_list - biases list;
#
# Output: list z_list - list of inputs to every layer;
#         list a_list - list of ouputs of every layer, acept last;
def FeedForward_class(X, W_list, b_list):

    # Initilize empty lists.
    z_list = []
    a_list = []

    # Compute first layer input and append it to the list.
    z_1 = X @ W_list[0] + b_list[0]
    z_list.append(z_1)

    # Compute first layer output and append it to the list for the chosen act function.
    a_1 = sigmoid(z_1)
    a_list.append(a_1)

    # Loop over all layers after first.
    for i in range(len(W_list) - 1):

        # Compute i-th layer input and append it to the list.
        z_i = a_list[i] @ W_list[i+1] + b_list[i+1]
        z_list.append(z_i)

        #
        # Note, we are computing activation function of the output layer using sigmoid.
        #
        # Compute i-th layer output and append it to the list for the chosen act function.
        a_i = sigmoid(z_i)
        a_list.append(a_i)
    
    return z_list, a_list





# This function perform Back Propagation step in order to train weights and biases.
#
# Input: np.array y_train - train target array;
#        np.array X_train - train x values array (or design matrix);
#        list W_list - weights list;
#        list b_list - biases list;
#        int M - mini-batch size;
#        int n_epoch - number of epochs;
#        double t_0 - decreasing learning rate parameter;
#        double t_1 - decreasing learning rate parameter.
#
# Output: list W_list - updated weights list;
#         list B_list - updated biases list.
def StochasticBackPropagation_class(y_train, X_train, W_list, b_list, M, n_epoch, t0, t1):
    
    n = X_train.shape[0] # Number of data points.

    m = int(n/M) # number of mini-batches.

    # Loop over epochs.
    for epoch in range(n_epoch):

        # Loop over mini-bathces
        for j in range(m):

            r = M * np.random.randint(m) # choosing random integer.

            # Making smaller arrays of initial ones.
            Xi = X_train[r : r + M]
            yi = y_train[r : r + M]

            # Make a FF step to get z_list and a_list for current min-batch.
            z_list, a_list = FeedForward_class(Xi, W_list, b_list)

            # Initialize list of deltas.
            delta_list = []

            #
            # Note! We are using another equation.
            #
            # Compute output layer delta and append it to the list.
            delta_out = (yi - a_list[-1]) * sigmoid_derivative(a_list[-1])
            delta_list.append(delta_out)

            # Loop over all layer.
            for i in range(len(W_list) - 1):

                # Compute i-th layer delta and append it to the list.
                delta_i = (delta_list[-1] @ (W_list[-1 - i]).T) * sigmoid_derivative(a_list[-1 - i])
                delta_list.append(delta_i)

            delta_list.reverse() # Revese deltas list.

            gamma = learning_schedule(epoch * m + j, t0, t1) # Compute decreasing lern rate at cur iter.

            # Update first layer weights and biases.
            W_list[0] += gamma * (Xi.T @ delta_list[0])
            b_list[0] += gamma * np.sum(delta_list[0])

            # Update i-th layer weights and biases.
            for i in range(len(W_list) - 1):
                W_list[i + 1] += gamma * (a_list[i].T @ delta_list[i + 1])
                b_list[i + 1] += gamma * np.sum(delta_list[i + 1])

    return W_list, b_list