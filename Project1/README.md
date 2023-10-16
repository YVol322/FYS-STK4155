# FYS-STK4155 Project 1: Comparing Linear, Ridge, and Lasso Regression on Franke's Function and Real Terrain Data

This project is an implemenation of Linear, Ridge and Lasso regression using Franke's function and real terrain data of Møsvatn Austfjell park in Norway as datasets. Also resampling techniques as bootstrap and cross-validation are studied.

## Code and results
Our code for the solution of the problems and figures have next structure:

- FYS-STK4155
    - Project1
        - Data
            - SRTM_data_Norway_2.tif
        - Figures
            - Bias_Variance
                - PDF
                    - PDF Figures
                - PNG
                    - PNG Figures
                - Frnake_plot
                    - PDF
                        - PDF Figures
                    - PNG
                        - PNG Figures
                - ...
        - Programs 
            - Bias_Variance.py
            - CrossVal.py
            - Franke_plot.py
            - Functions.py
            - Lasso.py
            - OLS.py
            - Ridge.py
            - RLO.py
            - Terrain_Bias_Variance.py
            - Terrain_CrossVal.py
            - Terrin_Data.py
            - Terrain_Lasso.py
            - Terrain_OLS.py
            - Terrain_Ridge.py
            - Terrain_RLO.py
            

To run the python programs, navigate to the directory of the program and run it by typing:
```bash
python OLS.py
```
or 
```bash
python3 OLS.py
```
for some operative systems.

You can notice that there are Bias_Variance.py, ... programs and Terrain_Bias_Variance.py programs. They are the same programs, just with different datasets. Since the programs are the same, comments are povided only for Bias_Variance.py types of pograms.

Suggestet order of running the programs and short desciption of them:
- Franke_plot.py - plots the Franke's function with and without noise.
- OLS.py - Linear regression on Franke's function.
- Ridge.py - Ridge regression on Franke's function.
- Lasso.py - Lasso regression on Franke's function.
- RLO.py - Comparison of there 3 regressions on Franke's function.
- Bias_Variance.py - Bootsrap method and bias-variance trade-off on Franke's function.
- CrossVal.py - Cross validation on Franke's function.
- Terrain_Data - plot the terrain data.
- Terrain_OLS.py - Linear regression on Terrain data.
- Terrain_Ridge.py - Ridge regression on Terrain data.
- Terrain_Lasso.py - Lasso regression on Terrain data.
- Terrain_RLO.py - Comparison of there 3 regressions on Terrain data.
- Terrain_Bias_Variance.py - Bootsrap method and bias-variance trade-off on Terrain data.
- Terrain_CrossVal.py - Cross validation on Terrain data.

Function.py can not be runned. It is just a program, that contains functions declarations.

PNG and PDF figures are the same, just in deifferent formats. This is done due to reason, that LaTex has better quality when working with pdf figures then png ones.
