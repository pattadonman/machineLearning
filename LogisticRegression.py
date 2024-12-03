import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class LogisticRegressionModel:
    def __init__(self):
        """
        Initialize the Logistic Regression model.
        """
        self.model = LogisticRegression()

    def fit(self, X, y):
        """
        Train the Logistic Regression model.

        Parameters:
        X: ndarray - Feature matrix.
        y: ndarray - Target labels.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict using the trained Logistic Regression model.

        Parameters:
        X: ndarray - Feature matrix for prediction.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities for each class.

        Parameters:
        X: ndarray - Feature matrix for prediction.
        """
        return self.model.predict_proba(X)

    def evaluate(self, y_true, y_pred):
        """
        Evaluate the model using accuracy, confusion matrix, and classification report.

        Parameters:
        y_true: ndarray - True labels.
        y_pred: ndarray - Predicted labels.
        """
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate synthetic dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    log_reg_model = LogisticRegressionModel()
    log_reg_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = log_reg_model.predict(X_test)

    # Evaluate the model
    log_reg_model.evaluate(y_test, y_pred)

    # Example: Predict probabilities for first 5 samples in the test set
    y_prob = log_reg_model.predict_proba(X_test[:5])
    print("\nPredicted Probabilities for first 5 samples:")
    print(pd.DataFrame(y_prob, columns=["Class 0", "Class 1"]))
