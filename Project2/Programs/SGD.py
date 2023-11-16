import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Functions import Data, SGD, SGD_Ada, SGD_RMS, SGD_ADAM, Create_dir

np.random.seed(2)

PNG_path, PDF_path = Create_dir('SGD')

x, y, X = Data()
X_train, X_test, x_train, x_test, y_train, y_test = train_test_split(X, y, x, test_size = 0.2)
n, degree = X_train.shape

Ms = np.array((5, 10, 20, 25, 50, 100))
n_epoch_array = np.array((2, 3, 4, 5, 10, 20))
t0s = np.arange(3, 8.1, 1)
M = 25
n_epoch = 5
t0, t1 = 5, 10
rho = 0.999
beta1 = 0.99
beta2 = 0.999
eps = 1e-4

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

i = 0
for M in Ms:
    beta = np.random.randn(degree,1)

    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        i += 1

        beta = SGD(X_train, y_train, degree, n, t0, t1, M, n_epoch)

    iter_GD_M.append(i)

    print(f"GD test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"minibatch size =  {M}")


for n_epoch in n_epoch_array:
    i = 0
    beta = np.random.randn(degree,1)

    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        i += 1

        beta = SGD(X_train, y_train, degree, n, t0, t1, M, n_epoch)

    iter_GD_n.append(i)

    print(f"GD test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"n_epochs =  {n_epoch}")


for t0 in t0s:
    i = 0
    beta = np.random.randn(degree,1)

    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        i += 1

        beta = SGD(X_train, y_train, degree, n, t0, t1, M, n_epoch)

    iter_GD_t.append(i)
    
    print(f"GD test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"t_0 = {t0}")



################################################ AdaGrad ################################################


for M in Ms:
    i = 0
    beta = np.random.randn(degree,1)

    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        i += 1

        beta = SGD_Ada(X_train, y_train, degree, n, t0, t1, M, n_epoch)

    iter_Ada_M.append(i)

    print(f"Ada test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"minibatch size =  {M}")


for n_epoch in n_epoch_array:
    i = 0
    beta = np.random.randn(degree,1)

    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        i += 1

        beta = SGD_Ada(X_train, y_train, degree, n, t0, t1, M, n_epoch)

    iter_Ada_n.append(i)

    print(f"Ada test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"n_epochs =  {n_epoch}")


for t0 in t0s:
    i = 0
    beta = np.random.randn(degree,1)

    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        i += 1

        beta = SGD_Ada(X_train, y_train, degree, n, t0, t1, M, n_epoch)

    iter_Ada_t.append(i)
    
    print(f"Ada test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"t_0 = {t0}")




################################################ RMSprop ################################################



for M in Ms:
    i = 0
    beta = np.random.randn(degree,1)

    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        i += 1

        beta = SGD_RMS(X_train, y_train, degree, n, rho, t0, t1, M, n_epoch)

    iter_RMS_M.append(i)

    print(f"RMS test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"minibatch size =  {M}")


for n_epoch in n_epoch_array:
    i = 0
    beta = np.random.randn(degree,1)

    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        i += 1

        beta = SGD_RMS(X_train, y_train, degree, n, rho, t0, t1, M, n_epoch)

    iter_RMS_n.append(i)

    print(f"RMS test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"n_epochs =  {n_epoch}")


for t0 in t0s:
    i = 0
    beta = np.random.randn(degree,1)

    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        i += 1

        beta = SGD_RMS(X_train, y_train, degree, n, rho, t0, t1, M, n_epoch)

    iter_RMS_t.append(i)
    
    print(f"RMS test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"t_0 = {t0}")
    


################################################ ADAM ################################################



for M in Ms:
    i = 0
    beta = np.random.randn(degree,1)

    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        i += 1

        beta = SGD_ADAM(X_train, y_train, degree, n, beta1, beta2, t0, t1, M, n_epoch)

    iter_ADAM_M.append(i)

    print(f"ADAM test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"minibatch size =  {M}")


for n_epoch in n_epoch_array:
    i = 0
    beta = np.random.randn(degree,1)

    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        i += 1

        beta = SGD_ADAM(X_train, y_train, degree, n, beta1, beta2, t0, t1, M, n_epoch)

    iter_ADAM_n.append(i)

    print(f"ADAM test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"n_epochs =  {n_epoch}")


for t0 in t0s:
    i = 0
    beta = np.random.randn(degree,1)

    while((mean_squared_error(X_train @ beta, y_train) > eps)):
        i += 1

        beta = SGD_ADAM(X_train, y_train, degree, n, beta1, beta2, t0, t1, M, n_epoch)

    iter_ADAM_t.append(i)
    
    print(f"ADAM test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"t_0 = {t0}")




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