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
python program2.py
```
or 
```bash
python3 program2.py
```
for some operative systems.

Tou can 

## Comment on the results
We have included the solution to our calculations in `.csv`-files. We have only included results up to `n_steps = 100 000`, because Github had a memory limit of `300Mb`. In our plotting code, we use results up to `n_steps = 1 000 000` and  `n_steps = 10 000 000`. To generate these, you will need to compile and run the c++ programs in which they are generated.

## The All folder
We have also decided to create a directory containing one C++ program and one Python program that solves all the tasks. The All directory has a structure:
- All

    - Thomas.cpp
    - Thomas.py
    - Results
        - Figures
            - figure1.pdf
            - ...
        - Tables
            - Analytical_solution
                - table1.csv
                - ...
            - General_Algorithm
                - table2.csv
                - ...
            - Special_Algorithm
                - table3.csv
                - ...

To run these programs, you can use the procedure from above while in the directory All.