import numpy as np
from Functions import FrankeFunction, Plot_Franke

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)
z_noise = FrankeFunction(x, y) + np.random.normal(0, 0.1, x.shape)

Plot_Franke(x,y,z, 'Franke')
Plot_Franke(x,y,z_noise, 'Franke_noise')

#z = z.reshape(-1,1)
#print(np.shape(z))
#
#
#n = 8
#X = create_X(x,y, n)
#
#z_train, z_test, X_train, X_test = train_test_split(z, X, test_size = 0.2)
#
#beta_OLS = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train)
#z_train_OLS = X_train @ beta_OLS
#z_test_OLS = X_test @ beta_OLS
#
#print(mean_squared_error(z_test, z_test_OLS))
#print(r2_score(z_test, z_test_OLS))
#print(mean_squared_error(z_train, z_train_OLS))
#print(r2_score(z_train, z_train_OLS))