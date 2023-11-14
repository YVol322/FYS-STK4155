import numpy as np
import matplotlib.pyplot as plt
from Functions import Data, GD

np.random.seed(2)

x, y, X, n, degree = Data()

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y

eps = 1e-5

etas = np.arange(0.1, 0.71, 0.1)

iters = []
for eta in etas:
    beta, i = GD(X, y, degree, n, eps, eta)
    iters.append(i)



plt.figure(1)
plt.plot(etas,iters)
plt.show()