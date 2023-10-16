import matplotlib.pyplot as plt
from pathlib import Path
from imageio import imread
from Functions import Create_directory


current_path = Path.cwd().resolve()
file_path = current_path.parent / 'Data' / 'SRTM_data_Norway_2.tif'
figures_path_PNG, figures_path_PDF = Create_directory('Terrain_Data') # Creating directory, to save figures to.


terrain1 = imread(file_path) # Reading the dataset.

# Plot of the terrain data.
plt.figure()
plt.title('Terrain over Norway')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(figures_path_PNG / "Terrain_Data")
plt.savefig(figures_path_PDF / "Terrain_Data", format = "pdf")
plt.show()