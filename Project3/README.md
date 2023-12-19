# FYS-STK4155 Project 3: Forward Euler and Deep Neural Network implementation for solving heat equation

This project is an implemenation of explicit Forward Euler (FE) and Deep Neural Network (DNN) methods for solving the heat equation with specific initial and boundary conditions.

## Code and results
All programs should be executed from "Project3" directory.
Our code for the solution of the problems and figures have next structure:

- FYS-STK4155
    - Project3
        - figures
            - PDF
                - PDF Figures
            - PNG
                - PNG Figures
        - programs 
            - analytical_solution.py
            - DNN_MSE_actfunctions.py
            - DNN_MSE_nodes_layers.py
            - DNN_MSE.py
            - DNN.py
            - forward_Euler_solution.py
            - functions.py
            

To run the python programs, navigate to the "Project3" directory and run selected program by typing:
```bash
python analytical_solution.py
```
or 
```bash
python3 analytical_solution.py
```
for some operative systems.

Suggestet order of running the programs and short desciption of them:
- functions.py - This program contains all functions that will be used in the following programs. Don't execute this program. You can, but it will do nothing.
- analytical_solution.py - This program computes analytical solution of the heat equation.
- forward_Euler_solution.py - This progam is implementation of FE algorithm for solving the heat equation.
- DNN.py - It is a DNN with 1 hidden layer and 250 hidden nodes implementation for solving the heat equation.
- DNN_MSE.py - This is a program that implements DNN for solving the heat equation and produces plot of mean squared error versus current iteration number.
- DNN_MSE_nodes_layers.py - This program is implementation of DNN with different number of hidden nodes and layers for solving the heat equation.
- DNN_MSE_actfunctions.py - This program is implementation of DNN for solving the heat equation using different activation functions. To change activation function it is needed to comment line 121 and uncomment line 125 or 126 in functions.py program.

PNG and PDF figures are the same, just in different formats. This is done due to reason, that LaTex has better quality when working with PDF figures then PNG ones.