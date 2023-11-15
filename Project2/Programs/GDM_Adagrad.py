import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Functions import Data, Ada_momentum, Create_dir

np.random.seed(2)

PNG_path, PDF_path = Create_dir('GD_momentum') # Creating directories to save figures to.

x, y, X = Data()
X_train, X_test, x_train, x_test, y_train, y_test = train_test_split(X, y, x, test_size = 0.2)
n, degree = X_train.shape

etas = np.arange(0.15, 0.85, 0.1).round(3)
momentums = np.arange(0.15, 0.85, 0.1).round(3)
eps = 1e-5

iters = np.zeros((etas.shape[0], momentums.shape[0]))
k, j = 0, 0
for eta in etas:
    for momentum in momentums:
        beta, i = Ada_momentum(X_train, y_train, degree, n, eps, eta, momentum)

        print(mean_squared_error(X_test @ beta, y_test))
        
        iters[k, j] = i
        k += 1
    k = 0
    j += 1

plt.figure()
sns.heatmap(iters, cmap="YlGnBu", annot=True, square=True, xticklabels = etas, yticklabels = momentums, fmt= '.0f')
plt.xlabel(r'$\gamma$')
plt.ylabel("momentum")
plt.savefig(PNG_path / 'GD_momentum_Ada_heatmap')
plt.savefig(PDF_path / 'GD_momentum_Ada_heatmap.pdf')
plt.show()