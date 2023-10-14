import numpy as np
from pathlib import Path
from Functions import FrankeFunction, Plot_Franke, Create_directory

N = 20

x = np.arange(0, 1, 1/N)
y = np.arange(0, 1, 1/N)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y)
z_noise = FrankeFunction(x, y) + np.random.normal(0, 0.1, np.shape(x))

current_path = Path.cwd().resolve()
print(current_path)

Create_directory('Franke_plot')

Plot_Franke(x,y,z, 'Franke')
Plot_Franke(x,y,z_noise, 'Franke_noise')