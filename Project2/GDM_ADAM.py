import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Functions import Data, ADAM_momentum

np.random.seed(2)

x, y, X, n, degree = Data()

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y

beta = np.random.randn(degree,1)

etas = np.arange(0.15, 0.85, 0.1).round(3)
momentums = np.arange(0.15, 0.85, 0.1).round(3)
eps = 1e-5
delta = 1e-7

beta1 = 0.99
beta2 = 0.999

iters = np.zeros((etas.shape[0], momentums.shape[0]))
k, j = 0, 0
for eta in etas:
    for momentum in momentums:
        beta, i = ADAM_momentum(X, y, degree, n, eps, eta, delta, beta1, beta2, momentum)
        
        iters[k, j] = i
        k += 1
    k = 0
    j += 1

plt.figure()
sns.heatmap(iters, cmap="YlGnBu", annot=True, square=True, xticklabels = momentums, yticklabels = etas, fmt= '.0f')
plt.xlabel("eta")
plt.ylabel("momentum")
plt.show()