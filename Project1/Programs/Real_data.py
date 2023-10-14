from imageio import imread
import matplotlib.pyplot as plt
from pathlib import Path
from Functions import Create_directory

Create_directory("Real_data")

current_path = Path.cwd().resolve()
file_path = current_path.parent / 'Data' / 'SRTM_data_Norway_2.tif'
figures_path_PNG = current_path.parent / "Figures" / "OLS" / "PNG"
figures_path_PDF = current_path.parent / "Figures" / "OLS" / "PDF"

# Load the terrain
terrain1 = imread(file_path)
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
#plt.savefig(figures_path_PNG / "Real_data")
#plt.savefig(figures_path_PDF / "Real_data", format = "pdf")
plt.show()