import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Functions import Data, SGD, SGD_Ada, SGD_RMS, SGD_ADAM, Create_dir

np.random.seed(2) # Setting the seed results can be repoduced.

PNG_path, PDF_path = Create_dir('SGD') # Creating directories to save figures to.

x, y, X = Data() # Generating data.
X_train, X_test, x_train, x_test, y_train, y_test = train_test_split(X, y, x, test_size = 0.2) # Splitting data.
n, degree = X_train.shape # Collecting number of data-points and features of design matrix.

Ms = np.array((5, 10, 20, 25, 50, 100)) # Array of mini-bath sizes.
n_epoch_array = np.array((2, 3, 4, 5, 10, 20)) # Array of number of epochs.
t0s = np.arange(30, 71, 10) # Array to t_0 constants.
M = 5 # Default mini-bath size.
n_epoch = 10 # Default number of epochs.
t0, t1 = 50, 100 # Defaul t_0 and t_1 constants.
rho = 0.999 # RMS parameter.
beta1 = 0.99 # ADAM parameter.
beta2 = 0.999 # ADAM parameter.
eps = 1e-5 # Stoping criterion.

# Initializing lists.
iter_GD_M = []
iter_Ada_M = []
iter_RMS_M = []
iter_ADAM_M = []
iter_GD_n = []
iter_Ada_n = []
iter_RMS_n = []
iter_ADAM_n = []
iter_GD_t = []
iter_Ada_t = []
iter_RMS_t = []
iter_ADAM_t = []


################################################ GD ################################################


# Loop over mini-batch sizes.
for M in Ms:
    i = 0 # Initilising number of iterations as zero.

    beta = np.random.randn(degree,1) # Initial guess.

    # Running loop until MSE is smaller then eps.
    while((mean_squared_error(X_train @ beta, y_train) > eps)):

        i += 1 # Adding one to iteration number.
        
        # SGD algo.
        beta = SGD(X_train, y_train, degree, n, t0, t1, M, n_epoch, beta)

    iter_GD_M.append(i) # Adding number of iterations utill convergence to the list.

    # Making sure MSE for test data is also small.
    print(f"GD test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"minibatch size =  {M}")


# Loop over number of epochs.
for n_epoch in n_epoch_array:
    i = 0 # Initilising number of iterations as zero.

    beta = np.random.randn(degree,1) # Initial guess.

    # Running loop until MSE is smaller then eps.
    while((mean_squared_error(X_train @ beta, y_train) > eps)):

        i += 1 # Adding one to iteration number.

        # SGD algo.
        beta = SGD(X_train, y_train, degree, n, t0, t1, M, n_epoch, beta)

    iter_GD_n.append(i) # Adding number of iterations utill convergence to the list.

    # Making sure MSE for test data is also small.
    print(f"GD test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"n_epochs =  {n_epoch}")


# Loop over t_0 constants.
for t0 in t0s:
    i = 0 # Initilising number of iterations as zero.

    beta = np.random.randn(degree,1) # Initial guess.

    # Running loop until MSE is smaller then eps.
    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        
        i += 1 # Adding one to iteration number.

        # SGD algo.
        beta = SGD(X_train, y_train, degree, n, t0, t1, M, n_epoch, beta)

    # Adding number of iterations utill convergence to the list.
    iter_GD_t.append(i)
    
    # Making sure MSE for test data is also small.
    print(f"GD test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"t_0 = {t0}")



################################################ AdaGrad ################################################


# Loop over mini-batch sizes.
for M in Ms:
    i = 0 # Initilising number of iterations as zero.

    beta = np.random.randn(degree,1) # Initial guess.

    # Running loop until MSE is smaller then eps.
    while((mean_squared_error(X_train @ beta, y_train) > eps)):

        i += 1 # Adding one to iteration number.

        # SGD Ada algo.
        beta = SGD_Ada(X_train, y_train, degree, n, t0, t1, M, n_epoch, beta)

    # Adding number of iterations utill convergence to the list.
    iter_Ada_M.append(i)

    # Making sure MSE for test data is also small.
    print(f"Ada test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"minibatch size =  {M}")



# Loop over number of epochs.
for n_epoch in n_epoch_array:
    i = 0 # Initilising number of iterations as zero.

    beta = np.random.randn(degree,1) # Initial guess.

    # Running loop until MSE is smaller then eps.
    while((mean_squared_error(X_train @ beta, y_train) > eps)):

        i += 1 # Adding one to iteration number.

        # SGD Ada algo.
        beta = SGD_Ada(X_train, y_train, degree, n, t0, t1, M, n_epoch, beta)

    # Adding number of iterations utill convergence to the list.
    iter_Ada_n.append(i)

    # Making sure MSE for test data is also small.
    print(f"Ada test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"n_epochs =  {n_epoch}")



# Loop over t_0 constants.
for t0 in t0s:
    i = 0 # Initilising number of iterations as zero.

    beta = np.random.randn(degree,1) # Initial guess.

    # Running loop until MSE is smaller then eps.
    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        
        i += 1 # Adding one to iteration number.

        # SGD Ada algo.
        beta = SGD_Ada(X_train, y_train, degree, n, t0, t1, M, n_epoch, beta)

    # Adding number of iterations utill convergence to the list.
    iter_Ada_t.append(i)
    
    # Making sure MSE for test data is also small.
    print(f"Ada test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"t_0 = {t0}")




################################################ RMSprop ################################################


# Loop over mini-batch sizes.
for M in Ms:
    i = 0 # Initilising number of iterations as zero.

    beta = np.random.randn(degree,1) # Initial guess.

    # Running loop until MSE is smaller then eps.
    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        
        i += 1 # Adding one to iteration number.

        # SGD RMS algo.
        beta = SGD_RMS(X_train, y_train, degree, n, rho, t0, t1, M, n_epoch, beta)

    # Adding number of iterations utill convergence to the list.
    iter_RMS_M.append(i)

    # Making sure MSE for test data is also small.
    print(f"RMS test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"minibatch size =  {M}")



# Loop over number of epochs.
for n_epoch in n_epoch_array:
    i = 0 # Initilising number of iterations as zero.

    beta = np.random.randn(degree,1) # Initial guess.

    # Running loop until MSE is smaller then eps.
    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        
        i += 1 # Adding one to iteration number.

        # SGD RMS algo.
        beta = SGD_RMS(X_train, y_train, degree, n, rho, t0, t1, M, n_epoch, beta)

    # Adding number of iterations utill convergence to the list.
    iter_RMS_n.append(i)

    # Making sure MSE for test data is also small.
    print(f"RMS test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"n_epochs =  {n_epoch}")



# Loop over t_0 constants.
for t0 in t0s:
    i = 0 # Initilising number of iterations as zero.

    beta = np.random.randn(degree,1) # Initial guess.

    # Running loop until MSE is smaller then eps.
    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        
        i += 1 # Adding one to iteration number.

        # SGD RMS algo.
        beta = SGD_RMS(X_train, y_train, degree, n, rho, t0, t1, M, n_epoch, beta)

    # Adding number of iterations utill convergence to the list.
    iter_RMS_t.append(i)
    
    # Making sure MSE for test data is also small.
    print(f"RMS test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"t_0 = {t0}")
    


################################################ ADAM ################################################


# Loop over mini-batch sizes.
for M in Ms:
    i = 0 # Initilising number of iterations as zero.

    beta = np.random.randn(degree,1) # Initial guess.

    # Running loop until MSE is smaller then eps.
    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        
        i += 1 # Adding one to iteration number.

        # SGD ADAM algo.
        beta = SGD_ADAM(X_train, y_train, degree, n, beta1, beta2, t0, t1, M, n_epoch, beta)

    # Adding number of iterations utill convergence to the list.
    iter_ADAM_M.append(i)

    # Making sure MSE for test data is also small.
    print(f"ADAM test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"minibatch size =  {M}")



# Loop over number of epochs.
for n_epoch in n_epoch_array:
    i = 0 # Initilising number of iterations as zero.

    beta = np.random.randn(degree,1) # Initial guess.

    # Running loop until MSE is smaller then eps.
    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        
        i += 1 # Adding one to iteration number.

        # SGD ADAM algo.
        beta = SGD_ADAM(X_train, y_train, degree, n, beta1, beta2, t0, t1, M, n_epoch, beta)

    # Adding number of iterations utill convergence to the list.
    iter_ADAM_n.append(i)

    # Making sure MSE for test data is also small.
    print(f"ADAM test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"n_epochs =  {n_epoch}")



# Loop over t_0 constants.
for t0 in t0s:
    i = 0 # Initilising number of iterations as zero.

    beta = np.random.randn(degree,1) # Initial guess.

    # Running loop until MSE is smaller then eps.
    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        
        i += 1 # Adding one to iteration number.

        # SGD ADAM algo.
        beta = SGD_ADAM(X_train, y_train, degree, n, beta1, beta2, t0, t1, M, n_epoch, beta)

    # Adding number of iterations utill convergence to the list.
    iter_ADAM_t.append(i)
    
    # Making sure MSE for test data is also small.
    print(f"ADAM test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"t_0 = {t0}")



# Generating some plots.

plt.figure(1)
plt.style.use('ggplot')
plt.plot(Ms, iter_GD_M, '-x', label = 'GD')
plt.plot(Ms, iter_Ada_M, '-x', label = 'Ada')
plt.plot(Ms, iter_RMS_M, '-x', label = 'RMS')
plt.plot(Ms, iter_ADAM_M, '-x', label = 'ADAM')
plt.legend()
plt.xlabel(r'$M$')
plt.ylabel(r'$N$')
plt.savefig(PNG_path / 'SGD_minibatch_iter')
plt.savefig(PDF_path / 'SGD_minibatch_iter.pdf')

plt.figure(2)
plt.style.use('ggplot')
plt.plot(n_epoch_array, iter_GD_n, '--x', label = 'GD')
plt.plot(n_epoch_array, iter_Ada_n, '--^', label = 'Ada')
plt.plot(n_epoch_array, iter_RMS_n, '--o', label = 'RMS')
plt.plot(n_epoch_array, iter_ADAM_n, '--s', label = 'ADAM')
plt.legend()
plt.xlabel(r'$n_{epochs}$')
plt.ylabel(r'$N$')
plt.savefig(PNG_path / 'SGD_nepochs_iter')
plt.savefig(PDF_path / 'SGD_nepochs_iter.pdf')


plt.figure(3)
plt.plot(t0s, iter_GD_t, '-x', label = 'GD')
plt.plot(t0s, iter_Ada_t, '-x', label = 'Ada')
plt.plot(t0s, iter_RMS_t, '-x', label = 'RMS')
plt.plot(t0s, iter_ADAM_t, '-x', label = 'ADAM')
plt.legend()
plt.xlabel(r'$t_0$')
plt.ylabel('$N$')
plt.savefig(PNG_path / 'SGD_gamma_iter')
plt.savefig(PDF_path / 'SGD_gamma_iter.pdf')

plt.figure(4)
plt.plot(n_epoch_array, iter_GD_n, '-x', label = 'GD')
plt.plot(n_epoch_array, iter_Ada_n, '-^', label = 'Ada')
plt.plot(n_epoch_array, iter_RMS_n, '-o', label = 'RMS')
plt.legend()
plt.xlabel(r'$n_{epochs}$')
plt.ylabel(r'$N$')
plt.savefig(PNG_path / 'SGD_nepochs_iter_ADAM-')
plt.savefig(PDF_path / 'SGD_nepochs_iter_ADAM-.pdf')
plt.show()