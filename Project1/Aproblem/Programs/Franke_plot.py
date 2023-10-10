import numpy as np
from Functions import FrankeFunction, Plot_Franke

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)
z_noise = FrankeFunction(x, y) + np.random.normal(0, 0.1, np.shape(x))

Plot_Franke(x,y,z, 'Franke')
Plot_Franke(x,y,z_noise, 'Franke_noise')