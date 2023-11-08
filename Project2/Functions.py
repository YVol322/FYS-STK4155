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

def grad(n, X, beta, y):
    gradient = (2.0/n)*X.T @ (X @ beta-y)

    return gradient

def GD_iter(n, X, y, beta, eta):
    gradient = grad(n, X, beta, y)
    beta -= eta*gradient

    return beta

def Ada(G, n, X, beta, y, eta, delta):
    gradient = grad(n, X, beta, y)

    G += (gradient @ gradient.T)
    G_diag = G.diagonal()
    sqrt_G = np.sqrt(G_diag)
    gamma = eta/(delta + sqrt_G).reshape(-1,1)

    return gamma

def RMS(G, rho, n, X, beta, y, eta, delta):
    gradient = grad(n, X, beta, y)

    G = (rho*G+(1-rho)*gradient*gradient)
    G_diag = G.diagonal()
    sqrt_G = np.sqrt(G_diag)
    gamma = eta/(delta + sqrt_G).reshape(-1,1)

    return gamma

def ADAM(beta1, beta2, n, X, beta, y, i, delta):
    first_moment = 0.0
    second_moment = 0.0
    
    gradient = grad(n, X, beta, y)

    first_moment = beta1*first_moment + (1-beta1)*gradient
    second_moment = beta2*second_moment+(1-beta2)*gradient*gradient
    first_term = first_moment/(1.0-beta1**(i+1))
    second_term = second_moment/(1.0-beta2**(i+1))

    update = first_term/((np.sqrt(second_term)+delta) * gradient)

    return update