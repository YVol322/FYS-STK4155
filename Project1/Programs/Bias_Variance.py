import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from Functions import Data, Create_directory, Create_X, Optimal_coefs_OLS, Prediction
from sklearn.utils import resample

np.random.seed(9)

N = 20 # Number of x and y points.
x,y,z = Data(N) # Generating the data.

maxdegree = 12
n_boostraps = 20 # numer of bootstraps.

polydegree = np.zeros(maxdegree)

MSE = np.zeros(maxdegree)
BIAS = np.zeros(maxdegree)
VAR = np.zeros(maxdegree)

figures_path_PNG, figures_path_PDF = Create_directory('Bias_Variance') # Creating directory, to save figures to.

for i in range(1, maxdegree+1, 1):

    X = Create_X(x, y, i) # Filling design matrix.

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2) # Train-test split of the data.

    z_pred = np.empty((z_test.shape[0], n_boostraps)) # Matrix with predictions in columns.


    for j in range(n_boostraps):
        X_, z_ = resample(X_train, z_train) # resampling train data.

        beta = Optimal_coefs_OLS(X_, z_) # OLS optimal coefs uing matrix inv.

        z_pred[:, j] = Prediction(X_test, beta).ravel() # Saving OLS test prediction to i-th col of matrix.

    # Computing MSE, BIAS and Variance and saving then to repectful arrays.
    MSE[i-1] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    BIAS[i-1] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    VAR[i-1] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
    
    polydegree[i-1] = i # Saving fit degree.

    # Printing errors to make sure MSE = BIAS + Variance.
    print('Error:', MSE[i-1])
    print('Bias^2:', BIAS[i-1])
    print('Var:', VAR[i-1])
    print('{} >= {} + {} = {}'.format(MSE[i-1], BIAS[i-1], VAR[i-1], VAR[i-1]+BIAS[i-1]))

# Plot of errors vs model complexity.
plt.figure()
plt.style.use('ggplot')
plt.plot(polydegree, MSE, label = 'MSE')
plt.plot(polydegree, BIAS, label = 'BIAS^2')
plt.plot(polydegree, VAR, label = 'Varianse')
plt.xlabel('Polynomial fit degree')
plt.ylabel('Error value')
plt.legend()
plt.savefig(figures_path_PNG / 'Bias_Variance')
plt.savefig(figures_path_PDF / 'Bias_Variance', format = 'pdf')
plt.show()