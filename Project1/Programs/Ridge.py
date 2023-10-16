import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Functions import Data, Create_directory, Create_X, Optimal_coefs_Ridge, Prediction


np.random.seed(2)

#N = 20
N = 200 # Number of x and y points.
x,y,z = Data(N) # Generating the data.

test_MSE_Ridge = []
train_MSE_Ridge = []
test_R2_Ridge = []
train_R2_Ridge = []

figures_path_PNG, figures_path_PDF = Create_directory('Ridge') # Creating directory, to save figures to.

n_lambdas = 100 # Number of penalty parameter lambda.
l = np.logspace(-3, 3, n_lambdas) # Log array of penalty parameters.
degree = 5 # Fit degree.

for lmbda in l:
    X = Create_X(x,y, degree) # Filling design matrix.

    z_train, z_test, X_train, X_test = train_test_split(z, X, test_size = 0.2) # Train-test split of the data.

    beta_Ridge = Optimal_coefs_Ridge(X_train, z_train, lmbda) # Ridge optimal coefs uing matrix inv.

    z_train_Ridge = Prediction(X_train, beta_Ridge) # Ridge train prediction.
    z_test_Ridge = Prediction(X_test, beta_Ridge) # Ridge test prediction.

    # Saving MSEs and R2 scores to lists.
    test_MSE_Ridge.append(mean_squared_error(z_test, z_test_Ridge))
    train_MSE_Ridge.append(mean_squared_error(z_train, z_train_Ridge))
    test_R2_Ridge.append(r2_score(z_test, z_test_Ridge))
    train_R2_Ridge.append(r2_score(z_train, z_train_Ridge))


# Plot of MSEs and R2 scores vs lambdas.
plt.figure(1)
plt.style.use('ggplot')
plt.subplot(2,1,1)
plt.plot(l, train_MSE_Ridge, label = 'Train MSE')
plt.plot(l, test_MSE_Ridge, label = 'Test MSE')
plt.xscale('log')
plt.ylabel('MSE')
plt.legend()
plt.subplot(2,1,2)
plt.plot(l, train_R2_Ridge, label = 'Train r2 score')
plt.plot(l, test_R2_Ridge, label = 'Test r2 score')
plt.xlabel('Penalty parameter')
plt.xscale('log')
plt.ylabel('R2 score')
plt.legend()
#plt.savefig(figures_path_PNG / 'Ridge_points20')
#plt.savefig(figures_path_PDF / 'Ridge_points20', format = 'pdf')
#plt.savefig(figures_path_PNG / 'Ridge_points200')
#plt.savefig(figures_path_PDF / 'Ridge_points200', format = 'pdf')
plt.show()