import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Functions import Data, SGDM, SGDM_Ada, SGDM_RMS, SGDM_ADAM, Create_dir

np.random.seed(2) # Setting the seed results can be repoduced.

PNG_path, PDF_path = Create_dir('SGD_momentum') # Creating directories to save figures to.

x, y, X = Data() # Generating data.
X_train, X_test, x_train, x_test, y_train, y_test = train_test_split(X, y, x, test_size = 0.2) # Splitting data.
n, degree = X_train.shape # Collecting number of data-points and features of design matrix.

Ms = np.array((10, 20, 25, 50)) # Array of mini-bath sizes.
momentums = np.arange(0.1, 0.65, 0.15).round(3) # Array of momentums.

t0, t1 = 50, 100 # Defaul t_0 and t_1 constants.
rho = 0.999 # RMS parameter.
beta1 = 0.99 # ADAM parameter.
beta2 = 0.999 # ADAM parameter.
eps = 1e-5 # Stoping criterion.
n_epoch = 10 # Default number of epochs.

# Initialising empty matrices.
map_GD = np.zeros((Ms.shape[0], momentums.shape[0]))
map_Ada = np.zeros((Ms.shape[0], momentums.shape[0]))
map_RMS = np.zeros((Ms.shape[0], momentums.shape[0]))
map_ADAM = np.zeros((Ms.shape[0], momentums.shape[0]))


################################################ GD ################################################

j, l = 0, 0 # Initial indices.

# Loop over momentums.
for momentum in momentums:

    # Loop over mini-batch sizes.
    for M in Ms:

        beta = np.random.randn(degree,1) # Initial guess.

        i = 0 # Initilising number of iterations as zero.

        # Running loop until MSE is smaller then eps.
        while((mean_squared_error(X_train @ beta, y_train) > eps)):
            
            i += 1 # Adding one to iteration number.

            # SGDM algo.
            beta = SGDM(X_train, y_train, degree, n, momentum, t0, t1, M, n_epoch, beta)

        # Making sure MSE for test data is also small.
        print(f"GD test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"minibatch size =  {M}", f"momentum =  {momentum}")

        map_GD[j,l] = i # Adding number of iterations utill convergence to the matrix.
        j += 1
    j = 0
    l += 1



################################################ AdaGrad ################################################


j, l = 0, 0 # Initial indices.

# Loop over momentums.
for momentum in momentums:

    # Loop over mini-batch sizes.
    for M in Ms:

        beta = np.random.randn(degree,1) # Initial guess.

        i = 0 # Initilising number of iterations as zero.

        # Running loop until MSE is smaller then eps.
        while((mean_squared_error(X_train @ beta, y_train) > eps)):
            
            i += 1 # Adding one to iteration number.

            # SGDM Ada algo.
            beta = SGDM_Ada(X_train, y_train, degree, n, momentum, t0, t1, M, n_epoch, beta)

        # Making sure MSE for test data is also small.
        print(f"Ada test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"minibatch size =  {M}", f"momentum =  {momentum}")

        map_Ada[j,l] = i # Adding number of iterations utill convergence to the matrix.
        j += 1
    j = 0
    l += 1



################################################ RMSProp ################################################



j, l = 0, 0 # Initial indices.

# Loop over momentums.
for momentum in momentums:

    # Loop over mini-batch sizes.
    for M in Ms:

        beta = np.random.randn(degree,1) # Initial guess.

        i = 0 # Initilising number of iterations as zero.

        # Running loop until MSE is smaller then eps.
        while((mean_squared_error(X_train @ beta, y_train) > eps)):
            
            i += 1 # Adding one to iteration number.

            # SGDM RMS algo.
            beta = SGDM_RMS(X_train, y_train, degree, n, rho, momentum, t0, t1, M, n_epoch, beta)

        # Making sure MSE for test data is also small.
        print(f"RMS test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"minibatch size =  {M}", f"momentum =  {momentum}")

        map_RMS[j,l] = i # Adding number of iterations utill convergence to the matrix.
        j += 1
    j = 0
    l += 1



################################################ ADAM ################################################



j, l = 0, 0 # Initial indices.

# Loop over momentums.
for momentum in momentums:

    # Loop over mini-batch sizes.
    for M in Ms:

        beta = np.random.randn(degree,1) # Initial guess.

        i = 0 # Initilising number of iterations as zero.

        # Running loop until MSE is smaller then eps.
        while((mean_squared_error(X_train @ beta, y_train) > eps)):
            
            i += 1 # Adding one to iteration number.

            # SGDM ADAM algo.
            beta = SGDM_ADAM(X_train, y_train, degree, n, beta1, beta2, momentum, t0, t1, M, n_epoch, beta)

        # Making sure MSE for test data is also small.
        print(f"ADAM test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"minibatch size =  {M}", f"momentum =  {momentum}")

        map_ADAM[j,l] = i # Adding number of iterations utill convergence to the matrix.
        j += 1
    j = 0
    l += 1



# Generating some plots.

plt.figure(1)
sns.heatmap(map_GD, cmap="YlGnBu", annot=True, square=True, xticklabels = momentums, yticklabels = Ms,fmt= '.0f')
plt.xlabel("Momentum")
plt.ylabel(r'$M$')
#plt.savefig(PNG_path / 'SGDM_heatmap')
#plt.savefig(PDF_path / 'SGDM_heatmap.pdf')



plt.figure(2)
sns.heatmap(map_Ada, cmap="YlGnBu", annot=True, square=True, xticklabels = momentums, yticklabels = Ms,fmt= '.0f')
plt.xlabel("Momentum")
plt.ylabel(r'$M$')
#plt.savefig(PNG_path / 'SGDM_Ada_heatmap')
#plt.savefig(PDF_path / 'SGDM_Ada_heatmap.pdf')


plt.figure(3)
sns.heatmap(map_RMS, cmap="YlGnBu", annot=True, square=True, xticklabels = momentums, yticklabels = Ms,fmt= '.0f')
plt.xlabel("Momentum")
plt.ylabel(r'$M$')
#plt.savefig(PNG_path / 'SGDM_RMS_heatmap')
#plt.savefig(PDF_path / 'SGDM_RMS_heatmap.pdf')


plt.figure(4)
sns.heatmap(map_ADAM, cmap="YlGnBu", annot=True, square=True, xticklabels = momentums, yticklabels = Ms,fmt= '.0f')
plt.xlabel("Momentum")
plt.ylabel(r'$M$')
#plt.savefig(PNG_path / 'SGDM_ADAM_heatmap')
#plt.savefig(PDF_path / 'SGDM_ADAM_heatmap.pdf')
plt.show()