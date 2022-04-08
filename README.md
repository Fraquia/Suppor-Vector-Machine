# Suppor-Vector-Machine
We developed our own Python Program for a nonlinear SVM with a Kernel.

# Authors 

This project was developed by:

1. Angelo Berardi (@AngeloBerardi)
2. Emanuele Fratocchi (@Fraquia)
3. Luca Urban 

# Description 

We built a  SVM classifier that distinguishes between images of handwritten digits from the MNIST database. Each sample is a 28x28 grayscale image, associated with a label. It is necessary to scale the data before the optimization.

We downloaded the full dataset but we extracted the target set made up of images belonging to number 3,8 and 6. We defined train and test  set by randomly splitting the original data with a percentage of 80 % and 20%. We also defined a k−fold cross validation to set
the hyperparameters. 

# SVM Algorithm

1. We wrote a program to find the solution of the SVM dual quadratic problem, identifying the values of the hyperparameters C and γ. We used specific method for Quadratic Programming problems to achieve better optimization performances.

2. Using the same kernel and hyperparameters of previous point we wrote a program which implements a decomposition method for the dual quadratic problem with any even value q ≥ 4. We defined the selection rule of the working set, constructed the subproblem at each iteration and use a standard algorithm for its solution. Again the use of specific method for quadratic programming to exploit the gradient of the objective function.

3. We fixed the dimension of the subproblem to q = 2 and implemented the most violating pair (MVP) decomposition method which uses the analytic solution of the subproblems.
