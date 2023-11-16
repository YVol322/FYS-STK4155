import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Functions import Data, GD, Ada, RMS, ADAM, Create_dir

np.random.seed(2)

PNG_path, PDF_path = Create_dir('GD') # Creating directories to save figures to.


x, y, X = Data() # Generating data.
X_train, X_test, x_train, x_test, y_train, y_test = train_test_split(X, y, x, test_size = 0.2) # Splitting data.
n, degree = X_train.shape # Collecting number of data-points and features of design matrix.


lmb = 0 # Penalty parameter. lmb = 0 means we are performing Linear regression analysis.
eps = 1e-5 # Stopping criterion.
etas = np.arange(0.1, 0.71, 0.1) # Array of learning rates.
lmbs = np.arange(0.0001, 0.0002, 0.00001) # Array of penalty parameters. We are performing Ridge regression analysis.
eta = 0.7 # Learning rate.
beta1 = 0.99 # ADAM parameter.
beta2 = 0.999 # ADAM parameter.
rho = 0.999 # RMS parameter.

# Initializing some list.
iters_GD_eta = []
iters_Ada_eta = []
iters_RMS_eta = []
iters_ADAM_eta = []
iters_GD_lmb = []
iters_Ada_lmb = []
iters_RMS_lmb = []
iters_ADAM_lmb = []
times_analyt = []
times_auto = []


################################################ GD ################################################

# GD using analytical gradients.

auto = 0 # This means we are using analytical expression to compute gradients.
for eta in etas:
    beta, i, time_1 = GD(X_train, y_train, degree, n, eps, eta, lmb, auto) # GD algo.

    iters_GD_eta.append(i) # Adding number of iterations utill convergence to the list.
    times_analyt.append(time_1) # Adding algo with analyt gradients runtime in seconds to the list.

    # Making sure MSE for test data is also small.
    print(f"GD test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"lmb =  {lmb},", f"eta = {eta}")
    

# GD using automatic differention gradients.

auto = 1 # Now we are using automatic differentiation to compute gradients.
for eta in etas:
    beta, i, time_2 = GD(X_train, y_train, degree, n, eps, eta, lmb, auto) 
    times_auto.append(time_2)# Adding algo with autodiff gradients runtime in seconds to the list.

    print(f"GD test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"lmb =  {lmb},", f"eta = {eta}")



eps = 1e-3 # Stopping criterion. Taking smaller criterion for faster MSE convergence.

auto = 0 # Using analyt expression for faster convergence. It just runs faster.
for lmb in lmbs:
    beta, i, t = GD(X_train, y_train, degree, n, eps, eta, lmb, auto)
    iters_GD_lmb.append(i)

    print(f"GD test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"lmb =  {lmb},", f"eta = {eta}")

# Converting lists to np arrays.
times_auto = np.array(times_auto)
times_analyt = np.array(times_analyt)


################################################ AdaGrad ################################################


for eta in etas:
    beta, i = Ada(X_train, y_train, degree, n, eps, eta, lmb)

    iters_Ada_eta.append(i)

    print(f"Ada test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"lmb =  {lmb},", f"eta = {eta}")



eps = 1e-3


for lmb in lmbs:
    beta, i = Ada(X_train, y_train, degree, n, eps, eta, lmb)

    iters_Ada_lmb.append(i)

    print(f"Ada test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"lmb =  {lmb},", f"eta = {eta}")



################################################ RMSprop ################################################


for eta in etas:
    beta, i = RMS(X_train, y_train, degree, n, eps, eta, rho, lmb)

    iters_RMS_eta.append(i)

    print(f"RMS test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"lmb =  {lmb},", f"eta = {eta}")


for lmb in lmbs:
    beta, i = RMS(X_train, y_train, degree, n, eps, eta, rho, lmb)

    iters_RMS_lmb.append(i)

    print(f"RMS test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"lmb =  {lmb},", f"eta = {eta}")



################################################ ADAM ################################################


for eta in etas:
    beta, i = ADAM(X_train, y_train, degree, n, eps, eta, beta1, beta2, lmb)

    iters_ADAM_eta.append(i)

    print(f"ADAM test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"lmb =  {lmb},", f"eta = {eta}")


for lmb in lmbs:
    beta, i = ADAM(X_train, y_train, degree, n, eps, eta, beta1, beta2, lmb)

    iters_ADAM_lmb.append(i)

    print(f"ADAM test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"lmb =  {lmb},", f"eta = {eta}")



# Generating some plots.

plt.figure(1)
plt.style.use('ggplot')
plt.plot(etas, iters_GD_eta, label = 'GD')
plt.plot(etas, iters_Ada_eta, label = 'AdaGrad')
plt.plot(etas, iters_RMS_eta, label = 'RMS')
plt.plot(etas, iters_ADAM_eta,  label = 'ADAM')
plt.xlabel(r'$\gamma$')
plt.ylabel('$N$')
plt.legend()
plt.savefig(PNG_path / 'GD_gamma_iter')
plt.savefig(PDF_path / 'GD_gamma_iter.pdf')


plt.figure(2)
plt.plot(lmbs, iters_GD_lmb, label = 'GD')
plt.plot(lmbs, iters_Ada_lmb, label = 'AdaGrad')
plt.plot(lmbs, iters_RMS_lmb, label = 'RMS')
plt.plot(lmbs, iters_ADAM_lmb, label = 'ADAM')
plt.xlabel(r'$\lambda$')
plt.ylabel('$N$')
plt.legend()
plt.savefig(PNG_path / 'GD_lambda_iter')
plt.savefig(PDF_path / 'GD_lambda_iter.pdf')


plt.figure(3)
plt.plot(etas, times_auto - times_analyt, '-x')
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\Delta t$, $s$')
plt.savefig(PNG_path / 'GD_gamma_time')
plt.savefig(PDF_path / 'GD_gamma_time.pdf')
plt.show()