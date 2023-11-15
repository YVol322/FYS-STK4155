import numpy as np
import matplotlib.pyplot as plt
from Functions import Data, SGD_ADAM

np.random.seed(2)

x, y, X, n, degree = Data()

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y

Ms = np.array((5, 10, 20, 25, 50, 100))
t0, t1 = 8000, 10000

n_epochs = []
eps = 1e-5
delta = 1e-7
beta1 = 0.99
beta2 = 0.999

for M in Ms:
    beta, i = SGD_ADAM(X, y, degree, n, eps, delta, beta1, beta2, t0, t1, M)
    n_epochs.append(i)
    

plt.figure(1)
plt.plot(Ms, n_epochs)
plt.show()