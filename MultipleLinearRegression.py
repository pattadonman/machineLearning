import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class MultipleLinearRegression:
    def __init__(self):
        """
        Initialize the Linear Regression model.
        """
        self.model = LinearRegression()
        self.coefficients = None
        self.intercept = None

    def generate_data(self, num_samples=100, num_features=3, random_state=42):
        """
        Generate synthetic data with multiple features.
        """
        np.random.seed(random_state)
        X = np.random.rand(num_samples, num_features)
        true_coefficients = np.random.randint(1, 10, size=num_features)
        y = X @ true_coefficients + np.random.randn(num_samples) * 0.5 + 10  # Linear equation + noise
        return X, y, true_coefficients

    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the Multiple Linear Regression model.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        return X_train, X_test, y_train, y_test

    def predict(self, X):
        """
        Predict the target values using the trained model.
        """
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred):
        """
        Evaluate the model's performance using MSE and R-squared.
        """
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, r2

    def visualize(self, y_test, y_pred, feature_name="Feature"):
        """
        Visualize predicted vs actual target values.
        """
        plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual")
        plt.scatter(range(len(y_pred)), y_pred, color="red", label="Prediction", alpha=0.7)
        plt.xlabel(f"Sample Index ({feature_name})")
        plt.ylabel("Target Value")
        plt.title("Actual vs Predicted")
        plt.legend()
        plt.show()

# Main script
if __name__ == "__main__":
    # Create an instance of the class
    regression = MultipleLinearRegression()
    
    # Generate data
    X, y, true_coefficients = regression.generate_data(num_samples=100, num_features=3)
    print(f"True Coefficients: {true_coefficients}")
    
    # Train the model
    X_train, X_test, y_train, y_test = regression.train(X, y)
    print(f"Learned Coefficients: {regression.coefficients}")
    print(f"Intercept: {regression.intercept:.2f}")
    
    # Predict
    y_pred = regression.predict(X_test)
    
    # Evaluate
    mse, r2 = regression.evaluate(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    
    # Visualize
    regression.visualize(y_test, y_pred, feature_name="Sample")
