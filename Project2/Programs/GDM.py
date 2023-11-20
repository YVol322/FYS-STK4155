import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Functions import Data, GD_momentum, Ada_momentum, RMS_momentum, ADAM_momentum, Create_dir

np.random.seed(2) # Setting the seed results can be repoduced.

PNG_path, PDF_path = Create_dir('GD_momentum') # Creating directories to save figures to.

x, y, X = Data() # Generating data.
X_train, X_test, x_train, x_test, y_train, y_test = train_test_split(X, y, x, test_size = 0.2) # Splitting data.
n, degree = X_train.shape # Collecting number of data-points and features of design matrix.

etas = np.arange(0.15, 0.85, 0.1).round(3) # Array of learning rates.
momentums = np.arange(0.15, 0.85, 0.1).round(3) # Array of momentums.
lmbs = np.arange(0.0001, 0.00017, 0.00001).round(5) # Array of penalty parameters.
momentum = 0.7 # Momentum.
lmb = 0 # Linear Regression penalty parameter.
eps = 1e-5 # Stopping criterion.
rho = 0.999 # RMS parameter.
beta1 = 0.99 # ADAM parameter
beta2 = 0.999 # ADAM parameter.

# Initialising empty matrices.
iters_GD_OLS = np.zeros((etas.shape[0], momentums.shape[0]))
iters_GD_Ridge = np.zeros((etas.shape[0], momentums.shape[0]))
iters_Ada = np.zeros((etas.shape[0], momentums.shape[0]))
iters_RMS = np.zeros((etas.shape[0], momentums.shape[0]))
iters_ADAM = np.zeros((etas.shape[0], momentums.shape[0]))


################################################ GD ################################################


k, j = 0, 0 # Initial indices.

# Loop over learning rates.
for eta in etas:

    # Loop over mometums.
    for momentum in momentums:

        # GMD algo.
        beta, i = GD_momentum(X_train, y_train, degree, n, eps, eta, momentum, lmb)

        # Making sure MSE for test data is also small.
        print(f"GD test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"eta = {eta}", f"moment = {momentum}")
        
        # Adding number of iteration to the matrix.
        iters_GD_OLS[k, j] = i
        k += 1
    k = 0
    j += 1


eps = 1e-3 # Stopping criterion. Taking smaller criterion for faster MSE convergence.

k, j = 0, 0 # Initial indices.

# Loop over learning rates.
for eta in etas:

    # Loop over penalty parameters.
    for lmb in lmbs:

        # GMD algo.
        beta, i = GD_momentum(X_train, y_train, degree, n, eps, eta, momentum, lmb)

        # Making sure MSE for test data is also small.
        print(f"GD test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"eta = {eta}", f"lmb = {lmb}")
        
        # Adding number of iteration to the matrix.
        iters_GD_Ridge[k, j] = i
        k += 1
    k = 0
    j += 1



################################################ AdaGrad ################################################

eps = 1e-5 # Changing back to 1e-5.

k, j = 0, 0 # Initial indices.

# Loop over learning rates.
for eta in etas:

    # Loop over mometums.
    for momentum in momentums:

        # Ada momentum algo.
        beta, i = Ada_momentum(X_train, y_train, degree, n, eps, eta, momentum)

        # Making sure MSE for test data is also small.
        print(f"Ada test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"eta = {eta}", f"moment = {momentum}")
        
        # Adding number of iteration to the matrix.
        iters_Ada[k, j] = i
        k += 1
    k = 0
    j += 1


################################################ RMSprop ################################################


k, j = 0, 0 # Initial indices.

# Loop over learning rates.
for eta in etas:

    # Loop over mometums.
    for momentum in momentums:

        # RMS momentum algo.
        beta, i = RMS_momentum(X_train, y_train, degree, n, eps, eta, rho, momentum)

        # Making sure MSE for test data is also small.
        print(f"RMS test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"eta = {eta}", f"moment = {momentum}")
        
        # Adding number of iteration to the matrix.
        iters_RMS[k, j] = i
        k += 1
    k = 0
    j += 1


################################################ ADAM ################################################


k, j = 0, 0 # Initial indices.

# Loop over learning rates.
for eta in etas:

    # Loop over mometums.
    for momentum in momentums:

        # RMS momentum algo.
        beta, i = ADAM_momentum(X_train, y_train, degree, n, eps, eta, beta1, beta2, momentum)

        # Making sure MSE for test data is also small.
        print(f"ADAM test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"eta = {eta}", f"moment = {momentum}")
        
        # Adding number of iteration to the matrix.
        iters_ADAM[k, j] = i
        k += 1
    k = 0
    j += 1


# Generating some plots.

plt.figure(1)
sns.heatmap(iters_GD_OLS, cmap="YlGnBu", annot=True, square=True, xticklabels = etas, yticklabels = momentums, fmt= '.0f')
plt.xlabel(r'$\gamma$')
plt.ylabel("momentum")
#plt.savefig(PNG_path / 'GD_momentum_heatmap')
#plt.savefig(PDF_path / 'GD_momentum_heatmap.pdf')

plt.figure(2)
sns.heatmap(iters_GD_Ridge, cmap="YlGnBu", annot=True, square=True, xticklabels = etas, yticklabels = lmbs, fmt= '.0f')
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\lambda$')
#plt.savefig(PNG_path / 'GD_momentum_Ridge_heatmap')
#plt.savefig(PDF_path / 'GD_momentum_Ridge_heatmap.pdf')

plt.figure(3)
sns.heatmap(iters_Ada, cmap="YlGnBu", annot=True, square=True, xticklabels = etas, yticklabels = momentums, fmt= '.0f')
plt.xlabel(r'$\gamma$')
plt.ylabel("momentum")
#plt.savefig(PNG_path / 'GD_momentum_Ada_heatmap')
#plt.savefig(PDF_path / 'GD_momentum_Ada_heatmap.pdf')

plt.figure(4)
sns.heatmap(iters_RMS, cmap="YlGnBu", annot=True, square=True, xticklabels = etas, yticklabels = momentums, fmt= '.0f')
plt.xlabel(r'$\gamma$')
plt.ylabel("momentum")
#plt.savefig(PNG_path / 'GD_momentum_RMS_heatmap')
#plt.savefig(PDF_path / 'GD_momentum_RMS_heatmap.pdf')

plt.figure(5)
sns.heatmap(iters_ADAM, cmap="YlGnBu", annot=True, square=True, xticklabels = etas, yticklabels = momentums, fmt= '.0f')
plt.xlabel(r'$\gamma$')
plt.ylabel("momentum")
#plt.savefig(PNG_path / 'GD_momentum_ADAM_heatmap')
#plt.savefig(PDF_path / 'GD_momentum_ADAM_heatmap.pdf')
plt.show()