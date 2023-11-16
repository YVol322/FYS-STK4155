import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Functions import Data, SGDM, SGDM_Ada, SGDM_RMS, SGDM_ADAM, Create_dir

np.random.seed(2)

PNG_path, PDF_path = Create_dir('SGD_momentum')

x, y, X = Data()
X_train, X_test, x_train, x_test, y_train, y_test = train_test_split(X, y, x, test_size = 0.2)
n, degree = X_train.shape

Ms = np.array((10, 20, 25, 50, 100))
momentums = np.arange(0.1, 0.8, 0.15).round(3)

t0, t1 = 5, 10
rho = 0.999
beta1 = 0.99
beta2 = 0.999
eps = 1e-4
n_epoch = 5

map_GD = np.zeros((Ms.shape[0], momentums.shape[0]))
map_Ada = np.zeros((Ms.shape[0], momentums.shape[0]))
map_RMS = np.zeros((Ms.shape[0], momentums.shape[0]))
map_ADAM = np.zeros((Ms.shape[0], momentums.shape[0]))

j, l = 0, 0
for momentum in momentums:
    for M in Ms:
        beta = np.random.randn(degree,1)

        i = 0
        while((mean_squared_error(X_train @ beta, y_train) > eps)):
            i += 1

            beta = SGDM(X_train, y_train, degree, n, momentum, t0, t1, M, n_epoch)

        print(f"GD test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"minibatch size =  {M}", f"momentum =  {momentum}")

        map_GD[j,l] = i
        j += 1
    j = 0
    l += 1




j, l = 0, 0
for momentum in momentums:
    for M in Ms:
        beta = np.random.randn(degree,1)

        i = 0
        while((mean_squared_error(X_train @ beta, y_train) > eps)):
            i += 1

            beta = SGDM_Ada(X_train, y_train, degree, n, momentum, t0, t1, M, n_epoch)

        print(f"Ada test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"minibatch size =  {M}", f"momentum =  {momentum}")

        map_Ada[j,l] = i
        j += 1
    j = 0
    l += 1




j, l = 0, 0
for momentum in momentums:
    for M in Ms:
        beta = np.random.randn(degree,1)

        i = 0
        while((mean_squared_error(X_train @ beta, y_train) > eps)):
            i += 1

            beta = SGDM_RMS(X_train, y_train, degree, n, rho, momentum, t0, t1, M, n_epoch)

        print(f"RMS test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"minibatch size =  {M}", f"momentum =  {momentum}")

        map_RMS[j,l] = i
        j += 1
    j = 0
    l += 1





j, l = 0, 0
for momentum in momentums:
    for M in Ms:
        beta = np.random.randn(degree,1)

        i = 0
        while((mean_squared_error(X_train @ beta, y_train) > eps)):
            i += 1

            beta = SGDM_ADAM(X_train, y_train, degree, n, beta1, beta2, momentum, t0, t1, M, n_epoch)

        print(f"ADAM test MSE: {mean_squared_error(X_test @ beta, y_test)},", f"minibatch size =  {M}", f"momentum =  {momentum}")

        map_ADAM[j,l] = i
        j += 1
    j = 0
    l += 1

plt.figure(1)
sns.heatmap(map_GD, cmap="YlGnBu", annot=True, square=True, xticklabels = momentums, yticklabels = Ms,fmt= '.0f')
plt.xlabel("Momentum")
plt.ylabel(r'$M$')
plt.savefig(PNG_path / 'SGDM_heatmap')
plt.savefig(PDF_path / 'SGDM_heatmap.pdf')



plt.figure(2)
sns.heatmap(map_Ada, cmap="YlGnBu", annot=True, square=True, xticklabels = momentums, yticklabels = Ms,fmt= '.0f')
plt.xlabel("Momentum")
plt.ylabel(r'$M$')
plt.savefig(PNG_path / 'SGDM_Ada_heatmap')
plt.savefig(PDF_path / 'SGDM_Ada_heatmap.pdf')


plt.figure(3)
sns.heatmap(map_RMS, cmap="YlGnBu", annot=True, square=True, xticklabels = momentums, yticklabels = Ms,fmt= '.0f')
plt.xlabel("Momentum")
plt.ylabel(r'$M$')
plt.savefig(PNG_path / 'SGDM_RMS_heatmap')
plt.savefig(PDF_path / 'SGDM_RMS_heatmap.pdf')


plt.figure(4)
sns.heatmap(map_ADAM, cmap="YlGnBu", annot=True, square=True, xticklabels = momentums, yticklabels = Ms,fmt= '.0f')
plt.xlabel("Momentum")
plt.ylabel(r'$M$')
plt.savefig(PNG_path / 'SGDM_ADAM_heatmap')
plt.savefig(PDF_path / 'SGDM_ADAM_heatmap.pdf')
plt.show()