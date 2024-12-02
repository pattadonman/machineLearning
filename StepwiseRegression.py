import statsmodels.api as sm
import pandas as pd
import numpy as np

class StepwiseRegression:
    def __init__(self, threshold_in=0.05, threshold_out=0.10):
        """
        Initialize Stepwise Regression model.

        Parameters:
        threshold_in: float - p-value threshold for adding predictors
        threshold_out: float - p-value threshold for removing predictors
        """
        self.threshold_in = threshold_in
        self.threshold_out = threshold_out
        self.selected_features = []
        self.final_model = None

    def fit(self, X, y):
        """
        Perform stepwise regression to select features.

        Parameters:
        X: DataFrame - Features
        y: Series/ndarray - Target variable
        """
        initial_features = list(X.columns)
        selected_features = []

        while True:
            changed = False

            # Forward step: Add variables
            remaining_features = list(set(initial_features) - set(selected_features))
            new_pval = pd.Series(index=remaining_features)
            for feature in remaining_features:
                model = sm.OLS(y, sm.add_constant(X[selected_features + [feature]])).fit()
                new_pval[feature] = model.pvalues[feature]
            if not new_pval.empty:
                min_pval = new_pval.min()
                if min_pval < self.threshold_in:
                    best_feature = new_pval.idxmin()
                    selected_features.append(best_feature)
                    changed = True
                    print(f"Added feature: {best_feature} (p-value: {min_pval:.4f})")

            # Backward step: Remove variables
            if selected_features:
                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                pvalues = model.pvalues.iloc[1:]  # Exclude intercept
                max_pval = pvalues.max()
                if max_pval > self.threshold_out:
                    worst_feature = pvalues.idxmax()
                    selected_features.remove(worst_feature)
                    changed = True
                    print(f"Removed feature: {worst_feature} (p-value: {max_pval:.4f})")

            if not changed:
                break

        self.selected_features = selected_features
        self.final_model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
        print("Stepwise regression completed.")
        print(f"Selected features: {self.selected_features}")

    def summary(self):
        """
        Return the summary of the final regression model.
        """
        if self.final_model is not None:
            return self.final_model.summary()
        else:
            return "No model has been fit yet."

    def predict(self, X):
        """
        Make predictions using the final model.
        """
        if self.final_model is None:
            raise ValueError("Model has not been trained yet.")
        return self.final_model.predict(sm.add_constant(X[self.selected_features]))

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X = pd.DataFrame({
        "X1": np.random.randn(100),
        "X2": np.random.randn(100),
        "X3": np.random.randn(100),
        "X4": np.random.randn(100)
    })
    y = 1.5 * X["X1"] + 2.0 * X["X3"] + np.random.randn(100) * 0.5

    # Perform stepwise regression
    stepwise_model = StepwiseRegression(threshold_in=0.05, threshold_out=0.10)
    stepwise_model.fit(X, y)
    print(stepwise_model.summary())

    # Make predictions
    predictions = stepwise_model.predict(X)
    print("Predictions:", predictions[:5])
