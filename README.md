# Breast Cancer Diagnosis via First-Order Optimization

This project explores the use of first-order optimization algorithms for binary classification through logistic regression, applied to the Breast Cancer Wisconsin (Diagnostic) dataset. The work investigates the convergence behavior of different gradient-based methods and the effect of L2 regularization on generalization performance.

## Objective

The goal is to predict whether a tumor is malignant or benign using logistic regression, optimized via gradient descent. The project emphasizes the mathematical foundation and practical implementation of first-order methods, with a focus on convergence behavior and regularization.

## Methodology

- Binary classification using logistic regression  
- Implementation of batch gradient descent with configurable step size  
- Analysis of convergence based on learning rate  
- Exploration of regularization effects  
- Evaluation using classification accuracy and cost function minimization  

## Regularization

Regularization is a technique used to prevent overfitting by penalizing large model weights. In this project, we use L2 regularization, which adds a term proportional to the squared norm of the weight vector to the loss function. This encourages the model to find simpler, more generalizable solutions. The strength of the regularization is controlled by a parameter \( \lambda \): higher values increase the penalty on large weights.

Mathematically, the regularized cost function becomes:
\[
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \log(1 + \exp(-y^{(i)} \theta^\top x^{(i)})) + \frac{\lambda}{2} \|\theta\|^2
\]

## Results

### Convergence of Optimization Algorithms

The plot below shows the decrease in the objective function \( f(\theta_k) \) over iterations \( k \) for both methods:

![Convergence of Optimization Methods](Convergence-study.png)

The accelerated gradient method (black) achieves faster convergence than standard gradient descent (red), reaching lower values of the loss function in fewer iterations.

### Effect of Regularization

The plot below shows how the misclassification rate varies with the regularization parameter \( \lambda \):

![Effect of Regularization](Regularization-effect.png)

Regularization helps control overfitting. The optimal classification accuracy is obtained for \( \lambda \approx 0.2 \), minimizing misclassification around 3.1%.
