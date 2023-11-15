import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Functions import Data, GD_momentum, Create_dir

np.random.seed(2)

PNG_path, PDF_path = Create_dir('GD_momentum') # Creating directories to save figures to.

x, y, X = Data()
X_train, X_test, x_train, x_test, y_train, y_test = train_test_split(X, y, x, test_size = 0.2)
n, degree = X_train.shape

etas = np.arange(0.15, 0.85, 0.1).round(3)
momentums = np.arange(0.15, 0.85, 0.1).round(3)
lmb = 0
eps = 1e-5

iters_OLS = np.zeros((etas.shape[0], momentums.shape[0]))
k, j = 0, 0
for eta in etas:
    for momentum in momentums:
        beta, i = GD_momentum(X_train, y_train, degree, n, eps, eta, momentum, lmb)

        print(mean_squared_error(X_test @ beta, y_test))
        
        iters_OLS[k, j] = i
        k += 1
    k = 0
    j += 1

etas = np.arange(0.15, 0.85, 0.1).round(3)
momentum = 0.7
lmbs = np.arange(0.0001, 0.00017, 0.00001).round(5)
eps = 1e-3

iters_Ridge = np.zeros((etas.shape[0], momentums.shape[0]))
k, j = 0, 0
for eta in etas:
    for lmb in lmbs:
        beta, i = GD_momentum(X_train, y_train, degree, n, eps, eta, momentum, lmb)

        print(mean_squared_error(X_test @ beta, y_test))
        
        iters_Ridge[k, j] = i
        k += 1
    k = 0
    j += 1


plt.figure(1)
sns.heatmap(iters_OLS, cmap="YlGnBu", annot=True, square=True, xticklabels = etas, yticklabels = momentums, fmt= '.0f')
plt.xlabel(r'$\gamma$')
plt.ylabel("momentum")
plt.savefig(PNG_path / 'GD_momentum_heatmap')
plt.savefig(PDF_path / 'GD_momentum_heatmap.pdf')

plt.figure(2)
sns.heatmap(iters_Ridge, cmap="YlGnBu", annot=True, square=True, xticklabels = etas, yticklabels = lmbs, fmt= '.0f')
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\lambda$')
plt.savefig(PNG_path / 'GD_momentum_Ridge_heatmap')
plt.savefig(PDF_path / 'GD_momentum_Ridge_heatmap.pdf')
plt.show()