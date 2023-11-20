# FYS-STK4155 Project 2: Gradient Descent and Feed Forward Neural Network implementation for Regression and Classification problem

This project is an implemenation of Gradient Descent (GD) methods, such as plain GD, GD with momentum, Stochastic Gradient Gescent (SGD) and SGD with momnetum on the simple second order polyomial regression problem. After GD we are studying Feed Forwad Neural Network (FFNN) implementation in order to solve the same regression ploblem and a classification ploblem using Wisconsin breast cancer dataset. In the end, we aslo develop a Logistic Regression to deal with the same breas cancer dataset.

## Code and results
All programs should be executed from "Project2/Programs" directory.
Our code for the solution of the problems and figures have next structure:

- FYS-STK4155
    - Project2
        - Figures
            - FFNN
                - PDF
                    - PDF Figures
                - PNG
                    - PNG Figures
            - FFNN_class
                - PDF
                    - PDF Figures
                - PNG
                    - PNG Figures
            - ...
        - Programs 
            - FFNN_Classification.py
            - FFNN_Leaky_RELU.py
            - FFNN_RELU.py
            - FFNN_Sigmoid.py
            - Functions.py
            - GD.py
            - GDGM.py
            - LogReg.py
            - Polynom.py
            - SGD.py
            - SGDM.py
            

To run the python programs, navigate to the directory of the program and run it by typing:
```bash
python OLS.py
```
or 
```bash
python3 OLS.py
```
for some operative systems.

Suggestet order of running the programs and short desciption of them:
- Functions.py - do not execute this program (you can, but it will not do anything), but take a look on how the algorithm are implemented. It is a good practise to keep it open and have a look on the function while analysing other programs.
- Polynom.py - Visualisation of dataset for Regression problem using simple 2nd order polynomial.
- GD.py - GD version of plain GD, AdaGrad, RMSProp and ADAM for Regression problem using simple 2nd order polynomial.
- GDGM.py - GD with momentum version of plain GD, AdaGrad, RMSProp and ADAM for Regression problem using simple 2nd order polynomial.
- SGD.py - SGD version of plain GD, AdaGrad, RMSProp and ADAM for Regression problem using simple 2nd order polynomial.
- SGDM.py - SGD with momentum version of plain GD, AdaGrad, RMSProp and ADAM for Regression problem using simple 2nd order polynomial.
- FFNN_Sigmoid.py - FFNN for Regression problem implementation using SGD algorithm with sigmoid as activation function using simple 2nd order polynomial.
- FFNN_RELU.py - FFNN for Regression problem implementation using SGD algorithm with RELU as activation function using simple 2nd order polynomial.
- FFNN_Leaky_RELU.py - FFNN for Regression problem implementation using SGD algorithm with leaky RELU as activation function using simple 2nd order polynomial.
- FFNN_Classification.py - FFNN for classification problem using SGD algorithm with sigmoid as activation function using Wisconsin breast cancer dataset.
- LogReg.py - Logistic Regression implementation using GD for Wisconsin breast cancer dataset.

Function.py can not be executed. It is just a program, that contains functions declarations.

PNG and PDF figures are the same, just in different formats. This is done due to reason, that LaTex has better quality when working with PDF figures then PNG ones.