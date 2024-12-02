import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class LassoRegressionModel:
    def __init__(self, alpha=1.0):
        """
        Initialize the Lasso Regression model with L1 regularization.
        """
        self.alpha = alpha  # Regularization strength
        self.model = Lasso(alpha=self.alpha)
        self.coefficients = None
        self.intercept = None

    def generate_data(self, num_samples=100, num_features=10, noise=0.5, random_state=42):
        """
        Generate synthetic data for regression.
        """
        np.random.seed(random_state)
        X = np.random.rand(num_samples, num_features)
        true_coefficients = np.random.rand(num_features)
        true_coefficients[true_coefficients < 0.5] = 0  # Make some coefficients zero for sparsity
        y = X @ true_coefficients + noise * np.random.randn(num_samples)
        return X, y, true_coefficients

    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the Lasso Regression model.
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
    # Create an instance of the LassoRegressionModel class
    lasso_model = LassoRegressionModel(alpha=0.1)
    
    # Generate data
    X, y, true_coefficients = lasso_model.generate_data(num_samples=100, num_features=10)
    print(f"True Coefficients: {true_coefficients}")
    
    # Train the model
    X_train, X_test, y_train, y_test = lasso_model.train(X, y)
    print(f"Learned Coefficients: {lasso_model.coefficients}")
    print(f"Intercept: {lasso_model.intercept}")
    
    # Predict
    y_pred = lasso_model.predict(X_test)
    
    # Evaluate
    mse, r2 = lasso_model.evaluate(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    
    # Visualize coefficients
    lasso_model.visualize_coefficients(true_coefficients)
