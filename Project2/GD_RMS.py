import numpy as np
import matplotlib.pyplot as plt
from Functions import Data, RMS

np.random.seed(2)

x, y, X, n, degree = Data()

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y

eps = 1e-5
delta = 1e-7
rho = 0.999

etas = np.arange(0.1, 0.71, 0.1)

iters = []
for eta in etas:
    beta, i = RMS(X, y, degree, n, eps, eta, delta, rho)
    print(beta[0])
    iters.append(i)

plt.figure(1)
plt.plot(etas,iters)
plt.show()