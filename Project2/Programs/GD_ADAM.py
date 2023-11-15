import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Functions import Data, ADAM, Create_dir

np.random.seed(2)

PNG_path, PDF_path = Create_dir('GD') # Using function just to get pathes to the directories.

x, y, X = Data()
X_train, X_test, x_train, x_test, y_train, y_test = train_test_split(X, y, x, test_size = 0.2)
n, degree = X_train.shape

eps = 1e-5
beta1 = 0.99
beta2 = 0.999
lmb = 0

etas = np.arange(0.1, 0.71, 0.1)

iters_eta = []
for eta in etas:
    beta, i = ADAM(X_train, y_train, degree, n, eps, eta, beta1, beta2, lmb)

    iters_eta.append(i)

    print(mean_squared_error(X_test @ beta, y_test))

lmbs = np.arange(0.0001, 0.0002, 0.00001)
eps = 1e-3
eta = 0.7

auto = 0
iters_lmb = []
for lmb in lmbs:
    beta, i = ADAM(X_train, y_train, degree, n, eps, eta, beta1, beta2, lmb)

    iters_lmb.append(i)

    print(mean_squared_error(X_test @ beta, y_test))


plt.figure(1)
plt.style.use('ggplot')
plt.plot(etas, iters_eta, '-x')
plt.xlabel(r'$\gamma$')
plt.ylabel('$N$')
plt.savefig(PNG_path / 'GD_ADAM_gamma_iter')
plt.savefig(PDF_path / 'GD_ADAM_gamma_iter.pdf')


plt.figure(2)
plt.plot(lmbs, iters_lmb, '-x')
plt.xlabel(r'$\lambda$')
plt.ylabel('$N$')
plt.savefig(PNG_path / 'GD_ADAM_lambda_iter')
plt.savefig(PDF_path / 'GD_ADAM_lambda_iter.pdf')
plt.show()