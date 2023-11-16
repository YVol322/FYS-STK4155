import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Functions import Data, GD_momentum, Ada_momentum, RMS_momentum, ADAM_momentum, Create_dir

np.random.seed(2)

PNG_path, PDF_path = Create_dir('GD_momentum') # Creating directories to save figures to.

x, y, X = Data()
X_train, X_test, x_train, x_test, y_train, y_test = train_test_split(X, y, x, test_size = 0.2)
n, degree = X_train.shape

etas = np.arange(0.15, 0.85, 0.1).round(3)
momentums = np.arange(0.15, 0.85, 0.1).round(3)
lmbs = np.arange(0.0001, 0.00017, 0.00001).round(5)
momentum = 0.7
lmb = 0
eps = 1e-5
rho = 0.999
beta1 = 0.99
beta2 = 0.999

iters_GD_OLS = np.zeros((etas.shape[0], momentums.shape[0]))
iters_GD_Ridge = np.zeros((etas.shape[0], momentums.shape[0]))
iters_Ada = np.zeros((etas.shape[0], momentums.shape[0]))
iters_RMS = np.zeros((etas.shape[0], momentums.shape[0]))
iters_ADAM = np.zeros((etas.shape[0], momentums.shape[0]))


################################################ GD ################################################


k, j = 0, 0
for eta in etas:
    for momentum in momentums:
        beta, i = GD_momentum(X_train, y_train, degree, n, eps, eta, momentum, lmb)

        print(f"GD test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"eta = {eta}", f"moment = {momentum}")
        
        iters_GD_OLS[k, j] = i
        k += 1
    k = 0
    j += 1


eps = 1e-3

k, j = 0, 0
for eta in etas:
    for lmb in lmbs:
        beta, i = GD_momentum(X_train, y_train, degree, n, eps, eta, momentum, lmb)

        print(f"GD test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"eta = {eta}", f"lmb = {lmb}")
        
        iters_GD_Ridge[k, j] = i
        k += 1
    k = 0
    j += 1



################################################ AdaGrad ################################################


k, j = 0, 0
for eta in etas:
    for momentum in momentums:
        beta, i = Ada_momentum(X_train, y_train, degree, n, eps, eta, momentum)

        print(f"Ada test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"eta = {eta}", f"moment = {momentum}")
        
        iters_Ada[k, j] = i
        k += 1
    k = 0
    j += 1


################################################ RMSprop ################################################


k, j = 0, 0
for eta in etas:
    for momentum in momentums:
        beta, i = RMS_momentum(X_train, y_train, degree, n, eps, eta, rho, momentum)

        print(f"RMS test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"eta = {eta}", f"moment = {momentum}")
        
        iters_RMS[k, j] = i
        k += 1
    k = 0
    j += 1


################################################ ADAM ################################################


k, j = 0, 0
for eta in etas:
    for momentum in momentums:
        beta, i = ADAM_momentum(X_train, y_train, degree, n, eps, eta, beta1, beta2, momentum)

        print(f"ADAM test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"eta = {eta}", f"moment = {momentum}")
        
        iters_ADAM[k, j] = i
        k += 1
    k = 0
    j += 1


plt.figure(1)
sns.heatmap(iters_GD_OLS, cmap="YlGnBu", annot=True, square=True, xticklabels = etas, yticklabels = momentums, fmt= '.0f')
plt.xlabel(r'$\gamma$')
plt.ylabel("momentum")
plt.savefig(PNG_path / 'GD_momentum_heatmap')
plt.savefig(PDF_path / 'GD_momentum_heatmap.pdf')

plt.figure(2)
sns.heatmap(iters_GD_Ridge, cmap="YlGnBu", annot=True, square=True, xticklabels = etas, yticklabels = lmbs, fmt= '.0f')
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\lambda$')
plt.savefig(PNG_path / 'GD_momentum_Ridge_heatmap')
plt.savefig(PDF_path / 'GD_momentum_Ridge_heatmap.pdf')

plt.figure(3)
sns.heatmap(iters_Ada, cmap="YlGnBu", annot=True, square=True, xticklabels = etas, yticklabels = momentums, fmt= '.0f')
plt.xlabel(r'$\gamma$')
plt.ylabel("momentum")
plt.savefig(PNG_path / 'GD_momentum_Ada_heatmap')
plt.savefig(PDF_path / 'GD_momentum_Ada_heatmap.pdf')

plt.figure(4)
sns.heatmap(iters_RMS, cmap="YlGnBu", annot=True, square=True, xticklabels = etas, yticklabels = momentums, fmt= '.0f')
plt.xlabel(r'$\gamma$')
plt.ylabel("momentum")
plt.savefig(PNG_path / 'GD_momentum_RMS_heatmap')
plt.savefig(PDF_path / 'GD_momentum_RMS_heatmap.pdf')

plt.figure(5)
sns.heatmap(iters_ADAM, cmap="YlGnBu", annot=True, square=True, xticklabels = etas, yticklabels = momentums, fmt= '.0f')
plt.xlabel(r'$\gamma$')
plt.ylabel("momentum")
plt.savefig(PNG_path / 'GD_momentum_ADAM_heatmap')
plt.savefig(PDF_path / 'GD_momentum_ADAM_heatmap.pdf')
plt.show()