import numpy as np
import matplotlib.pyplot as plt
from Functions import Data, ADAM

np.random.seed(2)

x, y, X, n, degree = Data()

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y

eps = 1e-5
delta = 1e-7
etas = np.arange(0.1, 0.71, 0.1)

beta1 = 0.99
beta2 = 0.999

iters = []
for eta in etas:
    beta, i = ADAM(X, y, degree, n, eps, eta, delta, beta1, beta2)
    iters.append(i)

plt.figure(1)
plt.plot(etas,iters)
plt.show()