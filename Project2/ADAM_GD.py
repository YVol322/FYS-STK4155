import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Functions import Data, GD_iter, ADAM

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
    i = 0

    beta = np.random.randn(degree,1)
    while(mean_squared_error(beta_linreg, beta)>eps):
        upd = ADAM(beta1, beta2, n, X, beta, y, i, delta)


        beta = GD_iter(n, X, y, beta, eta)
        i += 1
        
    print(eta)
    iters.append(i)

plt.figure(1)
plt.plot(etas,iters)
plt.show()