import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Functions import Data, Create_directory, Create_X, Optimal_coefs_OLS, Prediction


np.random.seed(3)

N = 20
#N = 200
x,y,z = Data(N)

test_MSE = []
train_MSE = []
test_R2 = []
train_R2 = []
betas = []
fit_degree = []

figures_path_PNG, figures_path_PDF = Create_directory('OLS')

maxdegree = 5

for degree in range(1,maxdegree + 1):
    X = Create_X(x,y, degree)

    z_train, z_test, X_train, X_test = train_test_split(z, X, test_size = 0.2)

    beta_OLS = Optimal_coefs_OLS(X_train, z_train)
    betas.append(beta_OLS)

    z_train_OLS = Prediction(X_train, beta_OLS)
    z_test_OLS = Prediction(X_test, beta_OLS)

    test_MSE.append(mean_squared_error(z_test, z_test_OLS))
    train_MSE.append(mean_squared_error(z_train, z_train_OLS))
    test_R2.append(r2_score(z_test, z_test_OLS))
    train_R2.append(r2_score(z_train, z_train_OLS))
    fit_degree.append(degree)

plt.figure(1)
plt.style.use('ggplot')
plt.subplot(2,1,1)
plt.plot(fit_degree, train_MSE, label = 'Train MSE')
plt.plot(fit_degree, test_MSE, label = 'Test MSE')
plt.ylabel('MSE')
plt.legend()
plt.subplot(2,1,2)
plt.plot(fit_degree, train_R2, label = 'Train r2 score')
plt.plot(fit_degree, test_R2, label = 'Test r2 score')
plt.xlabel('Polynomial fit degree')
plt.ylabel('R2 score')
plt.legend()
plt.savefig(figures_path_PNG / 'OLS_points20')
plt.savefig(figures_path_PDF / 'OLS_points20', format = 'pdf')
#plt.savefig(figures_path_PNG / 'OLS_points200')
#plt.savefig(figures_path_PDF / 'OLS_points200', format = 'pdf')
plt.show()

plt.figure(2)
for x in betas:
    plt.scatter(x, [fit_degree[betas.index(x)]] * len(x))

plt.xlabel('Optimal coefficients')
plt.ylabel('Polynomial fit degree')
plt.savefig(figures_path_PNG / 'OLS_betas')
plt.savefig(figures_path_PDF / 'OLS_betas', format = 'pdf')
plt.show()