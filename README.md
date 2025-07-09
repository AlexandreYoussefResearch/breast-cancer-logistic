# Breast Cancer Diagnosis via First-Order Optimization

This repository presents an implementation of logistic regression using first-order optimization methods to classify breast cancer tumors based on diagnostic features.

## Objective

The goal is to predict whether a tumor is malignant or benign using logistic regression, optimized via gradient descent. The project emphasizes the mathematical foundation and practical implementation of first-order methods, with a focus on convergence behavior and regularization.

## Dataset

- **Source:** [UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Number of instances:** 569
- **Features:** 30 numerical features computed from digitized images of fine needle aspirate (FNA) of breast masses
- **Target variable:** Diagnosis (`M` = malignant, `B` = benign)

## Methodology

- Binary classification using logistic regression
- Implementation of batch gradient descent with configurable step size
- Analysis of convergence based on learning rate
- Exploration of regularization effects
- Evaluation using classification accuracy and cost function minimization


## Results

- Converges to high classification accuracy (>95%) on the test set
- Learning rate analysis shows trade-off between speed and stability
- Regularization improves generalization for higher-dimensional inputs
- Visualization of decision boundary and cost function evolution provided

## Usage

Install the required dependencies:

```bash
pip install numpy pandas matplotlib
