from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

class RobustRegression:
    def __init__(self, epsilon=1.35, max_iter=100):
        """
        Initialize the Robust Regression model using Huber loss.

        Parameters:
        epsilon: float - The threshold at which the loss transitions from quadratic to linear.
        max_iter: int - Maximum number of iterations for the optimization.
        """
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.model = HuberRegressor(epsilon=self.epsilon, max_iter=self.max_iter)
        self.coefficients = None
        self.intercept = None

    def generate_data(self, num_samples=100, num_features=1, noise=1.0, outliers=0.1, random_state=42):
        """
        Generate synthetic data with outliers for regression.
        
        Parameters:
        num_samples: int - Number of data points.
        num_features: int - Number of features.
        noise: float - Standard deviation of Gaussian noise.
        outliers: float - Fraction of data points that are outliers.
        random_state: int - Random seed for reproducibility.
        """
        np.random.seed(random_state)
        X = np.random.rand(num_samples, num_features) * 10
        y = 3 * X[:, 0] + 5 + noise * np.random.randn(num_samples)
        
        # Add outliers
        num_outliers = int(num_samples * outliers)
        outlier_indices = np.random.choice(num_samples, num_outliers, replace=False)
        y[outlier_indices] += 20 * np.random.randn(num_outliers)  # Outlier magnitude
        
        return X, y

    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the Robust Regression model.
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
        Evaluate the model using MSE and R-squared metrics.
        """
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, r2

    def plot_results(self, X, y, y_pred):
        """
        Plot the original data and the regression line.
        """
        plt.scatter(X, y, color='blue', label='Data')
        plt.plot(X, y_pred, color='red', label='Robust Regression Line')
        plt.title("Robust Regression with Outliers")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.show()

# Main script
if __name__ == "__main__":
    # Create an instance of the RobustRegression class
    robust_model = RobustRegression(epsilon=1.35)

    # Generate data with outliers
    X, y = robust_model.generate_data(num_samples=100, outliers=0.15)
    
    # Train the model
    X_train, X_test, y_train, y_test = robust_model.train(X, y)
    print(f"Coefficients: {robust_model.coefficients}")
    print(f"Intercept: {robust_model.intercept}")

    # Predict
    y_pred_train = robust_model.predict(X_train)
    y_pred_test = robust_model.predict(X_test)

    # Evaluate
    mse_train, r2_train = robust_model.evaluate(y_train, y_pred_train)
    mse_test, r2_test = robust_model.evaluate(y_test, y_pred_test)
    print(f"Train MSE: {mse_train:.2f}, R-squared: {r2_train:.2f}")
    print(f"Test MSE: {mse_test:.2f}, R-squared: {r2_test:.2f}")

    # Plot results
    robust_model.plot_results(X, y, robust_model.predict(X))
