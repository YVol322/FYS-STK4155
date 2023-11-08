import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Functions import Data, GD_iter, Ada

np.random.seed(2)

x, y, X, n, degree = Data()

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y

eps = 1e-5
delta = 1e-7

etas = np.arange(0.1, 0.71, 0.1)

iters = []
for eta in etas:
    G = np.diag(np.zeros(degree))

    i = 0
    beta = np.random.randn(degree,1)
    while(mean_squared_error(beta_linreg, beta)>eps):
        gamma = Ada(G, n, X, beta, y, eta, delta)

        beta = GD_iter(n, X, y, beta, eta)
        i += 1
    print(eta)
    iters.append(i)

plt.figure(1)
plt.plot(etas,iters)
plt.show()