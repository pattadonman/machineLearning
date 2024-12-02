import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

class QuantileRegression:
    def __init__(self, quantile=0.5):
        """
        Initialize the Quantile Regression model.

        Parameters:
        quantile: float - Quantile to estimate (default is 0.5 for median regression).
        """
        self.quantile = quantile
        self.model = None
        self.results = None

    def fit(self, X, y):
        """
        Fit the Quantile Regression model.

        Parameters:
        X: DataFrame/ndarray - Features.
        y: Series/ndarray - Target variable.
        """
        X = sm.add_constant(X)  # Add intercept
        self.model = sm.QuantReg(y, X)
        self.results = self.model.fit(q=self.quantile)
        print(self.results.summary())

    def predict(self, X):
        """
        Predict using the trained Quantile Regression model.

        Parameters:
        X: DataFrame/ndarray - Features.
        """
        if self.results is None:
            raise ValueError("Model has not been trained yet.")
        X = sm.add_constant(X)  # Add intercept
        return self.results.predict(X)

    def plot_quantiles(self, X, y, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
        """
        Plot the regression lines for different quantiles.

        Parameters:
        X: DataFrame/ndarray - Features.
        y: Series/ndarray - Target variable.
        quantiles: list - List of quantiles to plot.
        """
        plt.scatter(X, y, color='blue', label='Data')

        for q in quantiles:
            model = sm.QuantReg(y, sm.add_constant(X)).fit(q=q)
            y_pred = model.predict(sm.add_constant(X))
            plt.plot(X, y_pred, label=f'Quantile {q}', linestyle='--')

        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Quantile Regression')
        plt.legend()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # Single feature
    y = 2 * X.squeeze() + 5 + np.random.randn(100) * 2  # Linear trend with noise

    # Introduce some outliers
    y[::10] += 15 * np.random.randn(10)

    # Initialize Quantile Regression model for the 50th percentile (median)
    qr_model = QuantileRegression(quantile=0.5)
    qr_model.fit(X, y)

    # Predict
    predictions = qr_model.predict(X)
    print(f"Predictions: {predictions[:5]}")

    # Plot quantiles
    qr_model.plot_quantiles(X, y, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])
