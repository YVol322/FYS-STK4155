import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Functions import Data, GD, Create_dir

np.random.seed(2)

PNG_path, PDF_path = Create_dir('GD') # Creating directories to save figures to.

x, y, X = Data() # Generating data.
X_train, X_test, x_train, x_test, y_train, y_test = train_test_split(X, y, x, test_size = 0.2) # Splitting data.
n, degree = X_train.shape # Collecting number of data-points and features of design matrix.

lmb = 0 # Penalty parameter. lmb = 0 means we are performing Linear regression analysis.
eps = 1e-5 # Stopping criterion.
etas = np.arange(0.1, 0.71, 0.1) # Array of learning rates.

# Initializing some list.
iters_eta = []
times_analyt = []
times_auto = []

auto = 0 # This means we are using analytical expression to compute gradients.
for eta in etas:
    beta, i, time_1 = GD(X_train, y_train, degree, n, eps, eta, lmb, auto) # GD algo.

    iters_eta.append(i) # Adding number of iterations utill convergence to the list.
    times_analyt.append(time_1) # Adding algo with analyt gradients runtime in seconds to the list.

    print(mean_squared_error(X_test @ beta, y_test)) # Making sure MSE for test data is also small.
    

auto = 1 # Now we are using automatic differentiation to compute gradients.
for eta in etas:
    beta, i, time_2 = GD(X_train, y_train, degree, n, eps, eta, lmb, auto) 
    times_auto.append(time_2)# Adding algo with autodiff gradients runtime in seconds to the list.

    print(mean_squared_error(X_test @ beta, y_test))




lmbs = np.arange(0.0001, 0.0002, 0.00001) # Array of penalty parameters. We are performing Ridge regression analysis.
eps = 1e-3 # Stopping criterion.
eta = 0.7 # Learning rate.

auto = 0 # Using analyt expression for faster convergence.
iters_lmb = []
for lmb in lmbs:
    beta, i, t = GD(X_train, y_train, degree, n, eps, eta, lmb, auto)
    iters_lmb.append(i)

    print(mean_squared_error(X_test @ beta, y_test))

# Converting lists to np arrays.
times_auto = np.array(times_auto)
times_analyt = np.array(times_analyt)

# Generating some plots.
plt.figure(1)
plt.style.use('ggplot')
plt.plot(etas, iters_eta, '-x')
plt.xlabel(r'$\gamma$')
plt.ylabel('$N$')
plt.savefig(PNG_path / 'GD_gamma_iter')
plt.savefig(PDF_path / 'GD_gamma_iter.pdf')


plt.figure(2)
plt.plot(lmbs, iters_lmb, '-x')
plt.xlabel(r'$\lambda$')
plt.ylabel('$N$')
plt.savefig(PNG_path / 'GD_lambda_iter')
plt.savefig(PDF_path / 'GD_lambda_iter.pdf')


plt.figure(3)
plt.plot(etas, times_auto - times_analyt, '-x')
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\Delta t$, $s$')
plt.savefig(PNG_path / 'GD_gamma_time')
plt.savefig(PDF_path / 'GD_gamma_time.pdf')
plt.show()