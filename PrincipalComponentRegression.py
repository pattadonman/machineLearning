import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class PrincipalComponentRegression:
    def __init__(self, n_components):
        """
        Initialize the PCR model.

        Parameters:
        n_components: int - Number of principal components to use.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.regressor = LinearRegression()
        self.X_pca = None

    def fit(self, X, y):
        """
        Fit the PCR model.

        Parameters:
        X: ndarray - Feature matrix.
        y: ndarray - Target variable.
        """
        # Perform PCA on the feature matrix
        self.X_pca = self.pca.fit_transform(X)

        # Train regression model on the principal components
        self.regressor.fit(self.X_pca, y)

    def predict(self, X):
        """
        Predict the target variable using the trained PCR model.

        Parameters:
        X: ndarray - Feature matrix for prediction.
        """
        # Transform features to principal components
        X_pca = self.pca.transform(X)
        return self.regressor.predict(X_pca)

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

    def explained_variance(self):
        """
        Get the explained variance ratio for each principal component.
        """
        return self.pca.explained_variance_ratio_

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 5) * 10  # 5 features
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + 5 + np.random.randn(100)  # Linear relationship with noise

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train PCR model
    n_components = 3  # Use 3 principal components
    pcr_model = PrincipalComponentRegression(n_components=n_components)
    pcr_model.fit(X_train, y_train)

    # Predict
    y_pred_train = pcr_model.predict(X_train)
    y_pred_test = pcr_model.predict(X_test)

    # Evaluate
    mse_train, r2_train = pcr_model.evaluate(y_train, y_pred_train)
    mse_test, r2_test = pcr_model.evaluate(y_test, y_pred_test)
    print(f"Train MSE: {mse_train:.2f}, R-squared: {r2_train:.2f}")
    print(f"Test MSE: {mse_test:.2f}, R-squared: {r2_test:.2f}")

    # Explained variance ratio
    print(f"Explained Variance Ratio: {pcr_model.explained_variance()}")

    # Plot the actual vs predicted values
    plt.scatter(y_test, y_pred_test, color='blue', label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Principal Component Regression')
    plt.legend()
    plt.show()
