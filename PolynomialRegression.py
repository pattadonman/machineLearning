import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class PolynomialRegression:
    def __init__(self, degree=2):
        """
        Initialize the Polynomial Regression model.

        Parameters:
        degree: int - The degree of the polynomial features.
        """
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=self.degree)
        self.model = LinearRegression()

    def fit(self, X, y):
        """
        Fit the Polynomial Regression model.

        Parameters:
        X: ndarray - Feature matrix.
        y: ndarray - Target variable.
        """
        X_poly = self.poly_features.fit_transform(X)
        self.model.fit(X_poly, y)

    def predict(self, X):
        """
        Predict the target variable using the trained Polynomial Regression model.

        Parameters:
        X: ndarray - Feature matrix for prediction.
        """
        X_poly = self.poly_features.transform(X)
        return self.model.predict(X_poly)

    def evaluate(self, y_true, y_pred):
        """
        Evaluate the model using MSE and R-squared.

        Parameters:
        y_true: ndarray - True target values.
        y_pred: ndarray - Predicted target values.
        """
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, r2

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # Single feature
    y = 2 * X**2 - 3 * X + 5 + np.random.randn(100, 1) * 5  # Quadratic relationship with noise

    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Polynomial Regression model
    degree = 2  # Quadratic regression
    poly_model = PolynomialRegression(degree=degree)
    poly_model.fit(X_train, y_train)

    # Predict
    y_pred_train = poly_model.predict(X_train)
    y_pred_test = poly_model.predict(X_test)

    # Evaluate
    mse_train, r2_train = poly_model.evaluate(y_train, y_pred_train)
    mse_test, r2_test = poly_model.evaluate(y_test, y_pred_test)
    print(f"Train MSE: {mse_train:.2f}, R-squared: {r2_train:.2f}")
    print(f"Test MSE: {mse_test:.2f}, R-squared: {r2_test:.2f}")

    # Plot the data and the polynomial regression curve
    X_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    y_pred_range = poly_model.predict(X_range)

    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X_range, y_pred_range, color='red', label=f'Polynomial Degree {degree}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Polynomial Regression')
    plt.legend()
    plt.show()
