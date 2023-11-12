import numpy as np
import matplotlib.pyplot as plt
from Functions import Data, GD, Ada, RMS, ADAM

np.random.seed(2)

x, y, X, n, degree = Data() #Generating the data.

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y # Linear regression optimal fit coefs.

eps = 1e-5 # Stoping crtirion.
delta = 1e-7 # Small paramenter for Ada, RMS and ADAM.
rho = 0.999 # Rho parameter for RMS.
beta1 = 0.99 # Beta1 parameter for ADAM.
beta2 = 0.999 # Beta2 parameter for ADAM.

etas = np.arange(0.1, 0.71, 0.1) # Array of learning rates.

iters_GD = []
iters_Ada = []
iters_RMS = []
iters_ADAM = []

for eta in etas:
    beta_GD, i_GD = GD(X, y, degree, n, eps, eta)
    print(i_GD)
    iters_GD.append(i_GD)

    beta_Ada, i_Ada = Ada(X, y, degree, n, eps, eta, delta)
    print(i_Ada)
    iters_Ada.append(i_Ada)

    beta_RMS, i_RMS = RMS(X, y, degree, n, eps, eta, delta, rho)
    print(i_RMS)
    iters_RMS.append(i_RMS)

    beta_ADAM, i_ADAM = ADAM(X, y, degree, n, eps, eta, delta, beta1, beta2)
    print(i_ADAM)
    iters_ADAM.append(i_ADAM)
    
    



plt.figure(1)
plt.plot(etas,iters_GD)
plt.plot(etas,iters_Ada)
plt.plot(etas,iters_RMS)
plt.plot(etas,iters_ADAM)
plt.show()