"""
Linear Regression from Scratch
================================

This module implements Linear Regression using only NumPy.
Perfect for understanding the fundamentals of ML!

Linear Regression finds the best-fit line: y = mx + b
- m (slope): How much y changes when x increases
- b (intercept): y value when x = 0

For multiple features: y = w1*x1 + w2*x2 + ... + b
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Linear Regression Model - Your First ML Algorithm!

    How it works:
    1. Start with random weights (slope) and bias (intercept)
    2. Make predictions with current weights
    3. Calculate error (how wrong are we?)
    4. Update weights to reduce error (gradient descent)
    5. Repeat steps 2-4 until error is small

    This is the foundation of ALL deep learning!
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the model

        Parameters:
        -----------
        learning_rate : float
            How big a step to take when updating weights.
            Too big = overshooting, too small = slow learning
            Think of it like step size when climbing down a hill

        n_iterations : int
            How many times to update the weights
            More iterations = better fit (but risk overfitting)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None  # Will store the slope(s)
        self.bias = None     # Will store the intercept
        self.loss_history = []  # Track learning progress

    def fit(self, X, y):
        """
        Train the model - Learn the best weights and bias!

        This is where the magic happens:
        - We use Gradient Descent to minimize error
        - Each iteration, we nudge weights toward better values

        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Training data (e.g., house sizes, number of rooms)
        y : numpy array, shape (n_samples,)
            Target values (e.g., house prices)
        """
        # Get dimensions
        n_samples, n_features = X.shape

        # Initialize weights and bias to zeros
        # Weights: one per feature (e.g., weight for size, weight for rooms)
        # Bias: single value (the intercept)
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent - The Learning Loop!
        for iteration in range(self.n_iterations):
            # --- STEP 1: Forward Pass (Make Predictions) ---
            # Formula: y_predicted = X * weights + bias
            # This is matrix multiplication: (n_samples, n_features) × (n_features,) = (n_samples,)
            y_predicted = np.dot(X, self.weights) + self.bias

            # --- STEP 2: Calculate Loss (How Wrong Are We?) ---
            # Mean Squared Error (MSE): average of squared differences
            # We square the errors so negative and positive errors don't cancel out
            loss = np.mean((y - y_predicted) ** 2)
            self.loss_history.append(loss)

            # --- STEP 3: Compute Gradients (Which Direction to Move?) ---
            # Gradient = derivative of loss with respect to weights/bias
            # This tells us: "If I change weight by small amount, how does loss change?"

            # Gradient for weights: -2/n * X^T * (y - y_predicted)
            # Shape: (n_features,)
            dw = -(2 / n_samples) * np.dot(X.T, (y - y_predicted))

            # Gradient for bias: -2/n * sum(y - y_predicted)
            # Shape: scalar
            db = -(2 / n_samples) * np.sum(y - y_predicted)

            # --- STEP 4: Update Parameters (Take a Step Down the Hill) ---
            # Move weights in opposite direction of gradient (descent!)
            # Learning rate controls step size
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print progress every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.4f}")

        print(f"\nTraining complete!")
        print(f"Final Loss: {self.loss_history[-1]:.4f}")
        print(f"Learned weights: {self.weights}")
        print(f"Learned bias: {self.bias:.4f}")

    def predict(self, X):
        """
        Make predictions on new data

        Once trained, use learned weights to predict new values

        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            New data to make predictions on

        Returns:
        --------
        predictions : numpy array, shape (n_samples,)
            Predicted values
        """
        # Same formula as in training: y = X * weights + bias
        return np.dot(X, self.weights) + self.bias

    def plot_loss(self):
        """
        Visualize the learning process

        A good loss curve should:
        - Start high (model is random)
        - Decrease steadily (model is learning)
        - Flatten out (model has converged)
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Mean Squared Error', fontsize=12)
        plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.show()

        # Check if model converged
        if len(self.loss_history) > 10:
            recent_change = abs(self.loss_history[-1] - self.loss_history[-10])
            if recent_change < 0.01:
                print("✓ Model converged (loss stopped decreasing)")
            else:
                print("⚠ Model may need more iterations or different learning rate")


def plot_predictions_1d(X, y, model, feature_name="Feature", target_name="Target"):
    """
    Visualize predictions for 1-feature regression

    Shows:
    - Actual data points (scatter)
    - Learned line (predictions)
    - How well the line fits

    Parameters:
    -----------
    X : numpy array, shape (n_samples, 1)
        Feature values
    y : numpy array, shape (n_samples,)
        True target values
    model : LinearRegression
        Trained model
    """
    plt.figure(figsize=(10, 6))

    # Plot actual data points
    plt.scatter(X, y, color='blue', alpha=0.6, s=50, label='Actual Data')

    # Plot prediction line
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, color='red', linewidth=2, label='Prediction Line')

    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel(target_name, fontsize=12)
    plt.title('Linear Regression: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.show()


def calculate_r2_score(y_true, y_pred):
    """
    Calculate R² (R-squared) score - How good is our model?

    R² measures how much variance in y is explained by X
    - R² = 1: Perfect predictions
    - R² = 0: Model is as good as predicting the mean
    - R² < 0: Model is worse than predicting the mean

    Formula: R² = 1 - (SS_residual / SS_total)
    - SS_residual: Sum of squared prediction errors
    - SS_total: Sum of squared deviations from mean

    Parameters:
    -----------
    y_true : numpy array
        Actual values
    y_pred : numpy array
        Predicted values

    Returns:
    --------
    r2 : float
        R² score between 0 and 1 (higher is better)
    """
    # Total sum of squares (variance in y)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)

    # Residual sum of squares (error in predictions)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    # R² score
    r2 = 1 - (ss_residual / ss_total)

    return r2


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("LINEAR REGRESSION FROM SCRATCH")
    print("=" * 60)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic data
    # Imagine: X = house size (sq ft), y = house price ($1000s)
    print("Generating synthetic data...")
    print("Scenario: Predicting house price from size")
    print()

    n_samples = 100
    X = 2.5 * np.random.randn(n_samples, 1) + 1.5  # House sizes (scaled)
    y = 3 * X.squeeze() + 2 + np.random.randn(n_samples) * 0.5  # True: y = 3x + 2 + noise

    print(f"Generated {n_samples} samples")
    print(f"X shape: {X.shape} (samples, features)")
    print(f"y shape: {y.shape} (samples,)")
    print(f"True relationship: y = 3*X + 2 (with noise)")
    print()

    # Create and train model
    print("-" * 60)
    print("Training Linear Regression Model...")
    print("-" * 60)
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    print()

    # Make predictions
    print("-" * 60)
    print("Making Predictions...")
    print("-" * 60)
    y_pred = model.predict(X)

    # Calculate R² score
    r2 = calculate_r2_score(y, y_pred)
    print(f"R² Score: {r2:.4f}")
    if r2 > 0.9:
        print("✓ Excellent fit!")
    elif r2 > 0.7:
        print("✓ Good fit")
    else:
        print("⚠ Model could be improved")
    print()

    # Compare true vs learned parameters
    print("-" * 60)
    print("Model Analysis:")
    print("-" * 60)
    print(f"True weight: 3.0")
    print(f"Learned weight: {model.weights[0]:.4f}")
    print(f"Error: {abs(3.0 - model.weights[0]):.4f}")
    print()
    print(f"True bias: 2.0")
    print(f"Learned bias: {model.bias:.4f}")
    print(f"Error: {abs(2.0 - model.bias):.4f}")
    print()

    # Visualize results
    print("Generating visualizations...")
    model.plot_loss()
    plot_predictions_1d(X, y, model, "House Size (scaled)", "House Price ($1000s)")

    print()
    print("=" * 60)
    print("CONGRATULATIONS! You've built your first ML model from scratch!")
    print("=" * 60)
