# Breast Cancer Diagnosis – Logistic Regression with First-Order Methods

This project implements and analyzes first-order optimization algorithms for binary classification using logistic regression. The objective is to identify whether a tumor is malignant or benign based on diagnostic features.

The analysis is based on the [Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), and includes both algorithmic convergence and regularization studies.

## Objective

The goal is to predict whether a tumor is malignant or benign using logistic regression, optimized via gradient descent. The project emphasizes the mathematical foundation and practical implementation of first-order methods, with a focus on convergence behavior and regularization.

## Problem Formulation

The logistic regression problem is cast as an unconstrained optimization task. The loss function is minimized over the training set to find the best decision boundary.

<p align="center">
  <img src="Problem.png" alt="Logistic Regression Objective Function" width="480"/>
</p>

## Methodology

- Binary classification using logistic regression
- Implementation of batch gradient descent with configurable step size
- Comparison of plain vs accelerated (Nesterov) gradient methods
- Exploration of L2 regularization (Ridge penalty)
- Evaluation using misclassification rate and loss convergence

## Results

### Convergence of Gradient vs Accelerated Methods

The figure below shows the convergence of the cost function with the number of iterations. The accelerated gradient method (black) reaches a lower cost significantly faster than standard gradient descent (red).

<p align="center">
  <img src="results/Convergence-study.png" alt="Convergence Plot" width="500"/>
</p>

### Effect of Regularization

Regularization helps prevent overfitting by penalizing large model weights. The plot below shows the effect of varying the regularization strength λ on the model's misclassification rate.

<p align="center">
  <img src="results/Regularization-effect.png" alt="Regularization Plot" width="500"/>
</p>

The optimal performance is observed for small values of λ, which balance bias and variance. As λ increases too much, underfitting leads to a degradation in accuracy.

## Contact

For any questions or collaborations, feel free to reach out:  
📧 alexandre.youssef.pro@gmail.com
