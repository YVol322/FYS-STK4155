import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Functions import Data, SGDM_ADAM

np.random.seed(2)

x, y, X, n, degree = Data()

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y

Ms = np.array((10, 20, 25, 50, 100))
t0, t1 = 500, 1000

eps = 1e-5
delta = 1e-7
beta1 = 0.99
beta2 = 0.999
momentums = np.arange(0.01, 0.14, 0.03).round(3)
change = 0

map = np.zeros((Ms.shape[0], momentums.shape[0]))
j, l = 0, 0
for momentum in momentums:
    for M in Ms:
        beta, i = SGDM_ADAM(X, y, degree, n, eps, delta, beta1, beta2, momentum, t0, t1, M)
        map[j,l] = i
        j += 1
    j = 0
    l += 1
    print(i, j)



plt.figure()
sns.heatmap(map, cmap="YlGnBu", annot=True, square=True, xticklabels = momentums, yticklabels = Ms,fmt= '.0f')
plt.xlabel("Momentum")
plt.ylabel("Minibatch size")
plt.show()