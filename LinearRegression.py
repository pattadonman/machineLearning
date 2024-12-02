#=======================================================================================
# """
# ===================================================
# Linear Regression Implementation with Visualization
# ===================================================

# Author       : Pattadon Nutes
# Date Created : December 2, 2024
# Last Updated : December 2, 2024
# Version      : 01.00.00
# Description  : 
#     This script demonstrates a simple implementation 
#     of Linear Regression using synthetic data. The 
#     model is trained with scikit-learn, and the results 
#     are visualized with Matplotlib.

# Python Version: 
# Dependencies  : 
#     - numpy 
#     - matplotlib 
#     - scikit-learn 

# Usage:
#     Run this script in a Python environment with 
#     the required libraries installed. It will 
#     generate a scatter plot showing the actual data 
#     points and the regression line.

# License:
#     
#=======================================================================================


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# generate fake datasets
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Feature
y = 4 + 3 * X + np.random.randn(100, 1)  # Target WITH noise

# spilt Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create and training model Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# predict results
y_pred = model.predict(X_test)

# calculate Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# display result
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Coefficient: {model.coef_[0][0]:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Visualization
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", label="Prediction")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression: Actual vs Prediction")
plt.legend()
plt.show()
