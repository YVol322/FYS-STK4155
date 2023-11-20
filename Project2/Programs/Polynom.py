import numpy as np
import matplotlib.pyplot as plt
from Functions import Data, Create_dir

np.random.seed(2) # Setting the seed results can be repoduced.

PNG_path, PDF_path = Create_dir('Polynomial') # Creating directories to save figures to.


x, y, X = Data() # Generating data.



# Generating a plot.


plt.figure(1)
plt.style.use('ggplot')
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
#plt.savefig(PNG_path / 'Polynomial')
#plt.savefig(PDF_path / 'Polynomial.pdf')
plt.show()