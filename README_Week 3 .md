# California Housing Price Prediction

# Project Overview
This notebook demonstrates various linear regression techniques to predict median house values in California districts using the California Housing Dataset. The primary objectives are to:

1.  Implement and evaluate simple and multi-feature linear regression models.
2.  Observe the performance of LASSO and Ridge regression in handling high-dimensional data with added noise.
3.  Understand the impact of the regularization parameter (`alpha`) on model performance and feature selection.

# Dataset
The project utilizes the California Housing Dataset, which contains various features for California housing districts. The goal is to predict the `MedHouseVal` (median house value, in $100,000s).

# 1. Environment Setup
The following libraries are imported for data manipulation, modeling, and visualization:
-   `pandas`
-   `numpy`
-   `matplotlib.pyplot`
-   `seaborn`
-   `sklearn.datasets.fetch_california_housing`
-   `sklearn.model_selection.train_test_split`
-   `sklearn.linear_model.LinearRegression`, `Lasso`, `Ridge`
-   `sklearn.metrics.mean_squared_error`, `r2_score`
-   `sklearn.preprocessing.StandardScaler`

The dataset is loaded, and a `MedHouseVal` column is added for the target variable. The dataset shape is `(20640, 9)`.

# 2. Simple Linear Regression (Income vs. House Value)
This section explores the linear relationship between 'Median Income' (`MedInc`) and 'Median House Value' (`MedHouseVal`).

-   **Feature Selection:** `X_simple = df[['MedInc']]`, `y = df['MedHouseVal']`
-   **Data Split:** Data is split into training and testing sets (`test_size=0.2`, `random_state=42`).
-   **Model Training:** A `LinearRegression` model is trained on `X_train`, `y_train`.
-   **Evaluation:** Predictions are made on `X_test`, and the results are visualized with a scatter plot and regression line.

## 3. Multi-Feature Linear Regression
This section builds a linear regression model using all available features to predict `MedHouseVal`.

-   **Scaling:** All features (excluding `MedHouseVal`) are scaled using `StandardScaler` to ensure they contribute equally to the model.
-   **Data Split:** The scaled data is split into training and testing sets.
-   **Model Training:** A `LinearRegression` model is trained on the scaled multi-feature data.
-   **Evaluation:** Predictions are made, and both R2 Score and Root Mean Squared Error (RMSE) are calculated.

This shows an improvement in predictive power compared to the simple linear regression model.

## 4. Regularized Models with High-Dimensional Noise
To simulate a high-dimensional scenario and test the robustness of regularization techniques, 20 random noise features are added to the scaled dataset.

-   **Adding Noise:** `noise = np.random.normal(0, 1, (len(X_scaled), 20))`, `X_noisy = np.hstack([X_scaled, noise])`
-   **Data Split:** The noisy data is split into training and testing sets.
-   **Model Comparison:** Three models are trained and evaluated:
    -   `LinearRegression` (No Penalty)
    -   `Ridge` (L2 Penalty, `alpha=10`)
    -   `Lasso` (L1 Penalty, `alpha=0.1`)

**Coefficient Visualization:** A plot is generated to visualize how each model handles coefficients, especially the shrinkage and sparsity induced by Lasso.

## 5. Alpha Parameter Tuning for Ridge and Lasso
This section further investigates the effect of the `alpha` regularization parameter on Ridge and Lasso models by testing a range of values (`[0.001, 1, 100]`).

