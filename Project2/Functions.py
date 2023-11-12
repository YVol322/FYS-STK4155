import numpy as np
from sklearn.metrics import mean_squared_error


def Data():
    n = 1000
    degree = 3
    x = np.arange(0, 1, 1/n).reshape(-1,1)
    y = 3 - 5 * x + 4 * x ** 2

    X = np.zeros((n, degree))

    for i in range(degree):
        X[:, i] = (x**i).ravel()

    return x, y, X, n, degree


def GD(X, y, degree, n, eps, eta):

    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)

    i=0
    while(mean_squared_error(beta_linreg, beta)>eps):
        gradient = (2.0/n)*X.T @ (X @ beta-y)

        beta -= eta*gradient

        i+=1
    
    return beta, i


def Ada(X, y, degree, n, eps, eta, delta):

    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)
    G = np.diag(np.zeros(degree))

    i=0
    while(mean_squared_error(beta_linreg, beta)>eps):
        gradient = (2.0/n)*X.T @ (X @ beta-y)

        G += gradient*gradient
        G_diag = G.diagonal()
        sqrt_G = np.sqrt(G_diag)

        gamma = eta/(delta + sqrt_G).reshape(-1,1)

        beta -= gamma*gradient

        i+=1


    return beta, i

def RMS(X, y, degree, n, eps, eta, delta, rho):
    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)
    G = np.diag(np.zeros(degree))

    i=0
    while(mean_squared_error(beta_linreg, beta)>eps):
        gradient = (2.0/n)*X.T @ (X @ beta-y)

        G = (rho*G+(1-rho)*gradient*gradient)
        G_diag = G.diagonal()
        sqrt_G = np.sqrt(G_diag)
        gamma = eta/(delta + sqrt_G).reshape(-1,1)

        beta -= gamma*gradient

        i+=1


    return beta, i

def ADAM(X, y, degree, n, eps, eta, delta, beta1, beta2):
    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)

    first_moment = 0.0
    second_moment = 0.0
    
    i=0
    while(mean_squared_error(beta_linreg, beta)>eps):
        gradient = (2.0/n)*X.T @ (X @ beta-y)

        first_moment = beta1*first_moment + (1-beta1)*gradient
        second_moment = beta2*second_moment+(1-beta2)*gradient*gradient
        first_term = first_moment/(1.0-beta1**(i+1))
        second_term = second_moment/(1.0-beta2**(i+1))

        update = eta*first_term/(np.sqrt(second_term)+delta)
        beta -= update
        
        i+=1


    return beta, i

def GD_momentum(X, y, degree, n, eps, eta, moment):

    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)

    change = 0

    i=0
    while(mean_squared_error(beta_linreg, beta)>eps):
        gradient = (2.0/n)*X.T @ (X @ beta-y)

        new_change = eta*gradient + moment * change

        beta -= new_change
        change = new_change

        i+=1
    
    return beta, i

def Ada_momentum(X, y, degree, n, eps, eta, delta, moment):

    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)
    G = np.diag(np.zeros(degree))

    change = 0

    i=0
    while(mean_squared_error(beta_linreg, beta)>eps):
        gradient = (2.0/n)*X.T @ (X @ beta-y)

        G += gradient*gradient
        G_diag = G.diagonal()
        sqrt_G = np.sqrt(G_diag)

        gamma = eta/(delta + sqrt_G).reshape(-1,1)

        new_change = gamma*gradient + moment * change

        beta -= new_change
        change = new_change

        i+=1

    return beta, i

def RMS_momentum(X, y, degree, n, eps, eta, delta, rho, moment):
    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)
    G = np.diag(np.zeros(degree))

    change = 0

    i=0
    while(mean_squared_error(beta_linreg, beta)>eps):
        gradient = (2.0/n)*X.T @ (X @ beta-y)

        G = (rho*G+(1-rho)*gradient*gradient)
        G_diag = G.diagonal()
        sqrt_G = np.sqrt(G_diag)
        gamma = eta/(delta + sqrt_G).reshape(-1,1)

        new_change = gamma*gradient + moment * change

        beta -= new_change
        print(beta[0])
        change = new_change

        i+=1

    return beta, i


def ADAM_momentum(X, y, degree, n, eps, eta, delta, beta1, beta2, moment):
    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    beta = np.random.randn(degree,1)

    first_moment = 0.0
    second_moment = 0.0

    change = 0
    
    i=0
    while(mean_squared_error(beta_linreg, beta)>eps):
        gradient = (2.0/n)*X.T @ (X @ beta-y)

        first_moment = beta1*first_moment + (1-beta1)*gradient
        second_moment = beta2*second_moment+(1-beta2)*gradient*gradient
        first_term = first_moment/(1.0-beta1**(i+1))
        second_term = second_moment/(1.0-beta2**(i+1))

        update = eta*first_term/(np.sqrt(second_term)+delta)

        new_change = update + moment * change

        beta -= new_change
        print(beta[0])
        change = new_change
        
        i+=1

    return beta, i