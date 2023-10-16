import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pathlib import Path
    

# This function computes Franke's function of given x and y arrays.
#
# Input: x - np.array of (N,N) shape with x values;
#        y - np.array of (N,N) shape with y values.
#
# Output: z - Franke's function with x and y values as input,
#         np.array of (N,N) shape with z values.
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4



# This function generates x and y arrays with given number of points, then uses them to compute z array.
#
# Input: N - number of x and y array points.
#
# Output: x,y - x and y np.arrays of (N,N) shape, z - Franke's function np.array of (N^2,1) shape.
def Data(N):
    x = np.arange(0, 1, 1/N) # array of shape (N,1).
    y = np.arange(0, 1, 1/N) # array of shape (N,1).
    x, y = np.meshgrid(x,y) # arrays of shape (N,N).

    z = FrankeFunction(x, y) + np.random.normal(0, 0.1, np.shape(x)) # array of shape (N,N).
    z = z.reshape(-1,1) # array of shape (N^2,1).

    return x,y,z


# This function generates takes terrain data as input, creating predictor values x and y, changing shape
# of terrain data and returns them.
#
# Input: terrain - imread(.tif file).
#        N - number of x,y and z array points.
#
# Output: x,y - x and y np.arrays of (N,N) shape, z - terrain data np.array of (N^2,1) shape.
def Terrain_Data(terrain, N):

    terrain_square = terrain[:N,:N] # Changing the shape of terrain data.

    x = np.linspace(0,1, np.shape(terrain_square)[0]) # Creating predictor values of N shape.
    y = np.linspace(0,1, np.shape(terrain_square)[1]) # Creating predictor values of N shape.
    x, y = np.meshgrid(x,y) # (N,N) arrays.

    z = terrain_square
    z = z.reshape(-1,1) # (N^2, 1) array.

    return x,y,z



# This function creates desing function with given degree n and x and y arrays.
#
# Input: x - np.array of (N,N) or (N,1) shape with x values;
#        y - np.array of (N,N) or (N,1) shape with y values;
#        n - polynomial fit degree.
#
# Ouptut: X - design matrix of (n, int((n+1) * (n+2)/2)) shape.
def Create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta.
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X



# This function detects current directory path and crates directories, in wich .png and .pdf 
# figures will be saved.
#
# Input: string with name of directories that will be crated.
#
# Otpur: void, creates directories.
def Create_directory(path):
    current_path = Path.cwd().resolve()

    figures_path = current_path.parent / "Figures" / path
    figures_path.mkdir(parents=True, exist_ok=True)

    figures_path_PNG = current_path.parent / "Figures" / path / "PNG"
    figures_path_PNG.mkdir(parents=True, exist_ok=True)

    figures_path_PDF = current_path.parent / "Figures" / path / "PDF"
    figures_path_PDF.mkdir(parents=True, exist_ok=True)

    return figures_path_PNG, figures_path_PDF



# This function plots and saves Franke's fuction of given x, y, z arrays and plots them.
#
# Input: x - np.array of (N,N) shape with x values;
#        y - np.array of (N,N) shape with y values;
#        z - np.array of (N^2,N^2) shape with z values.
#
# Output: void, plots and saves figures of Franke's fucntion.

def Plot_Franke(x, y, z, name):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    current_path = Path.cwd().resolve()
    figures_path = current_path.parent / "Figures" / "Franke_plot"

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(figures_path / 'PDF' / name, format = 'pdf')
    plt.savefig(figures_path / 'PNG' / name)
    plt.show()



# This function find OLS optimal fit coefs via matrix inverse.
#
# Input: X_train - train data design matrix of of (0.8 *n, 0.8 * int((n+1) * (n+2)/2)) shape;
#        z_train - train data of (0.8 * N^2, 1) shape.
#
# Output: beta_OLS - np.array of (int((n+1) * (n+2)/2), 1) shape - optimal fit coefs.
def Optimal_coefs_OLS(X_train, z_train):
    beta_OLS = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train)

    return beta_OLS



# This functions uses result of Optimal_coefs_OLS function to compute model prediction.
#
# Input - X - train or test design matrix (shape depends of input values);
# beta - optimal fit coefs (shape depends of input values).
#
# Output - z_OLS - model prediction (shape depends of input values).
def Prediction(X, beta):
    z_OLS = X @ beta

    return z_OLS



# This function find Ridge optimal fit coefs via matrix inverse.
#
# Input: X_train - train data design matrix of of (0.8 *n, 0.8 * int((n+1) * (n+2)/2)) shape;
#        z_train - train data of (0.8 * N^2, 1) shape.
#
# Output: beta_Ridge - np.array of (int((n+1) * (n+2)/2), 1) shape - optimal fit coefs.
def Optimal_coefs_Ridge(X_train, z_train, l):
    I = np.eye(np.shape(X_train.T.dot(X_train))[0])
    beta_Ridge = np.linalg.pinv(X_train.T.dot(X_train) + l * I).dot(X_train.T).dot(z_train)

    return beta_Ridge