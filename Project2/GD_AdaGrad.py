import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Functions import Data, Ada

np.random.seed(2)

x, y, X, n, degree = Data()

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y

eps = 1e-5
delta = 1e-7

etas = np.arange(0.1, 0.71, 0.1)

iters = []
for eta in etas:
    beta, i = Ada(X, y, degree, n, eps, eta, delta)
    print(beta)
    iters.append(i)

plt.figure(1)
plt.plot(etas,iters)
plt.show()