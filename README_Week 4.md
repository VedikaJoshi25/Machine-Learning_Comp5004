# COMP5004 Machine Learning Notebook: Churn Prediction and SVM Introduction

This notebook explores fundamental machine learning concepts through two main parts:

## Part 1: Logistic Regression for Churn Prediction
This section demonstrates the process of building and evaluating a logistic regression model for customer churn prediction using a synthetic dataset.

### 1. Setup and Data Generation
-   Generates a synthetic dataset of telecom customers including `tenure`, `monthly_charges`, and `churn` status.

### 2. Pre-processing
-   Splits the data into training and testing sets.
-   Standardizes features (`tenure` and `monthly_charges`) using `StandardScaler`, which is crucial for logistic regression's performance.

### 3. Model Implementation
-   Trains a Logistic Regression model on the scaled training data.
-   Extracts and displays prediction probabilities for the test set.

### 4. Decision Threshold Challenge
-   Introduces the concept of a decision threshold in logistic regression.
-   Provides a function `evaluate_threshold` to visualize the confusion matrix for a given threshold.
-   Demonstrates the confusion matrix for the default threshold (0.5).

### 5. Additional Questions
-   **Classification Report**: Generates a detailed classification report for the default threshold (0.5).
-   **Threshold Adjustment Discussion**: Explains when and why to adjust the decision threshold, particularly focusing on the trade-off between precision and recall when the cost of false negatives is high.
-   **Adjusted Threshold Evaluation**: Adjusts the threshold to 0.3 and presents the new classification report, highlighting the impact on recall for the churn class.
-   **Summary of Threshold Impact**: Provides a concise comparison of model performance metrics (precision, recall, accuracy) between the default and adjusted thresholds.

## Part 2: Support Vector Machines (SVM) Introduction
This section introduces Support Vector Machines, focusing on visualizing decision boundaries and the effect of key hyperparameters.

### 1. Setup: Visualizing the Margin
-   Loads the Breast Cancer dataset and uses only two features ('mean radius', 'mean texture') for easy visualization.
-   Scales the data, which is critical for SVM performance.

### 2. Linear SVM and the C parameter
-   Illustrates the concept of the 'soft margin' in linear SVMs.
-   Compares decision boundaries and support vectors for different `C` parameter values (0.01 and 100), showing how `C` controls the penalty for misclassification.

### 3. The Kernel Trick (Non-Linear)
-   Demonstrates the use of the Radial Basis Function (RBF) kernel to create non-linear decision boundaries.
-   Visualizes an RBF kernel SVM with `C=1` and `gamma=0.7`.

### Modify RBF SVM with gamma=10
-   Creates and visualizes a new RBF SVM model with `gamma=10`, showcasing the effect of a higher gamma value on the decision boundary (making it more complex).

### SVM Accuracy Comparison
-   Calculates and compares the accuracy of a Linear SVM (C=100) and two RBF SVMs (C=1, gamma=0.7 and C=1, gamma=10) on the scaled training data.
