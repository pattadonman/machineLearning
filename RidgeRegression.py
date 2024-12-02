import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class RidgeRegressionModel:
    def __init__(self, alpha=1.0):
        """
        Initialize the Ridge Regression model with L2 regularization.
        """
        self.alpha = alpha  # Regularization strength
        self.model = Ridge(alpha=self.alpha)
        self.coefficients = None
        self.intercept = None

    def generate_data(self, num_samples=100, num_features=10, noise=0.5, random_state=42):
        """
        Generate synthetic data for regression.
        """
        np.random.seed(random_state)
        X = np.random.rand(num_samples, num_features)
        true_coefficients = np.random.rand(num_features)
        y = X @ true_coefficients + noise * np.random.randn(num_samples)
        return X, y, true_coefficients

    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the Ridge Regression model.
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

    def visualize_coefficients(self, true_coefficients):
        """
        Visualize true vs learned coefficients.
        """
        plt.plot(range(len(true_coefficients)), true_coefficients, 'bo-', label='True Coefficients')
        plt.plot(range(len(self.coefficients)), self.coefficients, 'ro-', label='Learned Coefficients')
        plt.title("True vs Learned Coefficients")
        plt.xlabel("Feature Index")
        plt.ylabel("Coefficient Value")
        plt.legend()
        plt.show()

# Main script
if __name__ == "__main__":
    # Create an instance of the RidgeRegressionModel class
    ridge_model = RidgeRegressionModel(alpha=1.0)
    
    # Generate data
    X, y, true_coefficients = ridge_model.generate_data(num_samples=100, num_features=10)
    print(f"True Coefficients: {true_coefficients}")
    
    # Train the model
    X_train, X_test, y_train, y_test = ridge_model.train(X, y)
    print(f"Learned Coefficients: {ridge_model.coefficients}")
    print(f"Intercept: {ridge_model.intercept}")
    
    # Predict
    y_pred = ridge_model.predict(X_test)
    
    # Evaluate
    mse, r2 = ridge_model.evaluate(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    
    # Visualize coefficients
    ridge_model.visualize_coefficients(true_coefficients)
