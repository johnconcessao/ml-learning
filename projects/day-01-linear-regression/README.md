# Day 1 Project: Linear Regression from Scratch

## Overview
A complete implementation of Linear Regression using only NumPy. This project demonstrates the fundamental concepts of machine learning: training, prediction, and evaluation.

## What You'll Learn

### Core ML Concepts
1. **Gradient Descent**: How models actually learn
2. **Loss Functions**: Measuring model performance (MSE)
3. **Model Training**: Iterative parameter optimization
4. **Prediction**: Using learned parameters on new data
5. **Evaluation**: R² score and performance metrics

### Mathematical Foundation
- Linear equations: y = wx + b
- Matrix multiplication: Forward pass
- Derivatives: Computing gradients
- Optimization: Minimizing loss

## Project Structure

```
day-01-linear-regression/
├── src/
│   └── linear_regression.py      # Complete implementation
├── linear_regression_tutorial.ipynb  # Interactive tutorial
├── data/                          # (Generated synthetically)
├── visualizations/                # (Generated during execution)
└── README.md                      # This file
```

## Quick Start

### Option 1: Python Script
```bash
cd projects/day-01-linear-regression
python src/linear_regression.py
```

### Option 2: Jupyter Notebook (Recommended)
```bash
cd projects/day-01-linear-regression
jupyter notebook linear_regression_tutorial.ipynb
```

## Code Walkthrough

### 1. Model Initialization
```python
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
```
- **learning_rate**: Step size for weight updates (0.01 = safe default)
- **n_iterations**: Number of training steps (1000 = usually enough)

### 2. Training (fit method)
```python
model.fit(X, y)
```

**What happens inside:**
```python
# Step 1: Make predictions
y_pred = X @ weights + bias  # Matrix multiplication

# Step 2: Calculate error
loss = mean((y - y_pred)²)   # Mean Squared Error

# Step 3: Compute gradients (calculus!)
dw = -2/n * X.T @ (y - y_pred)  # How to change weights
db = -2/n * sum(y - y_pred)      # How to change bias

# Step 4: Update parameters
weights -= learning_rate * dw    # Gradient descent step
bias -= learning_rate * db
```

**Repeat 1000 times** → weights converge to optimal values!

### 3. Prediction
```python
y_pred = model.predict(X_new)
```
Simply: `y = X @ learned_weights + learned_bias`

### 4. Evaluation
```python
r2 = calculate_r2_score(y_true, y_pred)
```
R² = 1 - (prediction_error / variance_in_data)
- R² = 1.0: Perfect predictions
- R² = 0.0: As good as predicting the mean
- R² < 0.0: Worse than predicting the mean (very bad!)

## Key Code Blocks Explained

### Block 1: Forward Pass (Prediction)
```python
y_predicted = np.dot(X, self.weights) + self.bias
```
**What it does**:
- Multiply each feature by its weight
- Add them up (dot product)
- Add the bias term
- This is a LINEAR combination → "Linear" Regression

**Example** (2 features):
```
X = [house_size=1500, num_rooms=3]
weights = [0.2, 100]
bias = 50000

price = 1500*0.2 + 3*100 + 50000 = 50,600
```

### Block 2: Loss Calculation
```python
loss = np.mean((y - y_predicted) ** 2)
```
**What it does**:
- Calculate prediction error: (actual - predicted)
- Square it (so negative errors don't cancel positive ones)
- Take the mean (average over all samples)

**Why square?**
- Penalizes large errors more (error of 10 is worse than 10 errors of 1)
- Mathematically convenient (derivative is linear)

### Block 3: Gradient Computation
```python
dw = -(2 / n_samples) * np.dot(X.T, (y - y_predicted))
db = -(2 / n_samples) * np.sum(y - y_predicted)
```
**What it does**:
- Calculates the derivative of loss with respect to weights
- Derivative tells us: "In which direction should I change the weight to reduce loss?"
- Negative gradient points downhill (toward lower loss)

**Intuition**:
- If prediction > actual: decrease weight
- If prediction < actual: increase weight
- Magnitude tells us how much to change

### Block 4: Parameter Update
```python
self.weights -= self.learning_rate * dw
self.bias -= self.learning_rate * db
```
**What it does**:
- Move weights in the direction that reduces loss
- Learning rate controls how big a step we take

**Why subtract?**
- Gradient points uphill (toward higher loss)
- We want to go downhill (toward lower loss)
- So we go in the OPPOSITE direction (subtract)

## Performance Metrics

### Mean Squared Error (MSE)
- **What**: Average squared difference between predictions and actual values
- **Range**: 0 to ∞ (lower is better)
- **Use**: Training objective (what we minimize)

### R² Score (Coefficient of Determination)
- **What**: Proportion of variance explained by the model
- **Range**: -∞ to 1 (higher is better)
- **Interpretation**:
  - 1.0: Perfect predictions
  - 0.9: Explains 90% of variance (excellent)
  - 0.5: Explains 50% of variance (okay)
  - 0.0: No better than predicting the mean
  - < 0: Worse than predicting the mean

## How Gradient Descent Works

Imagine you're on a mountain (loss surface) in fog:
1. You want to reach the bottom (minimum loss)
2. You can only see your immediate surroundings (gradient)
3. You feel which direction is downhill (negative gradient)
4. You take a step downhill (update weights)
5. Repeat until you reach the bottom (convergence)

**Learning rate = step size:**
- Too large: You might overshoot the bottom
- Too small: Takes forever to reach the bottom
- Just right: Efficient descent

## Common Issues & Solutions

### Problem: Loss is increasing
**Cause**: Learning rate too high (taking too big steps)
**Solution**: Decrease learning_rate (try 0.001 instead of 0.01)

### Problem: Loss decreasing very slowly
**Cause**: Learning rate too low OR need more iterations
**Solution**: Increase learning_rate OR increase n_iterations

### Problem: Loss plateaus but is still high
**Cause**: Model too simple for data (linear can't fit non-linear)
**Solution**: Try polynomial features or neural network

### Problem: Perfect training, poor test performance
**Cause**: Overfitting (not relevant for linear regression usually)
**Solution**: More data or regularization (Ridge/Lasso)

## Experiments to Try

### 1. Effect of Learning Rate
```python
# Too high
model_high = LinearRegression(learning_rate=0.5, n_iterations=1000)

# Too low
model_low = LinearRegression(learning_rate=0.0001, n_iterations=1000)

# Compare loss curves!
```

### 2. Multiple Features
```python
# House: size, rooms, age
X = np.random.randn(100, 3)
y = 2*X[:,0] + 3*X[:,1] - 0.5*X[:,2] + 5

model = LinearRegression()
model.fit(X, y)
# Should learn weights ≈ [2, 3, -0.5] and bias ≈ 5
```

### 3. Non-linear Data
```python
# Create non-linear relationship
X = np.random.randn(100, 1)
y = X**2 + noise  # Quadratic!

# Linear model will fit poorly (low R²)
# Try adding X² as a feature!
X_poly = np.hstack([X, X**2])
```

## Mathematical Derivations

### MSE Loss
```
L = (1/n) Σ(y_i - ŷ_i)²
```

### Gradient of MSE w.r.t weights
```
∂L/∂w = -(2/n) Σ x_i(y_i - ŷ_i)
      = -(2/n) X^T (y - ŷ)
```

### Gradient of MSE w.r.t bias
```
∂L/∂b = -(2/n) Σ(y_i - ŷ_i)
```

### Update Rules
```
w_new = w_old - α * ∂L/∂w
b_new = b_old - α * ∂L/∂b
```
where α = learning_rate

## Connection to Neural Networks

Linear Regression is a **1-layer neural network** with:
- No activation function (or identity activation)
- MSE loss
- Gradient descent optimization

**To get a neural network**, just add:
1. More layers: y = W₃(W₂(W₁x + b₁) + b₂) + b₃
2. Non-linear activations: y = W₃·ReLU(W₂·ReLU(W₁x))
3. Same gradient descent, but with backpropagation!

## Real-World Applications

1. **Housing Prices**: Size, location → price
2. **Sales Forecasting**: Advertising spend → sales
3. **Medical**: Height, weight → blood pressure
4. **Finance**: Interest rates → stock prices
5. **Baseline Model**: Always try linear regression first!

## Next Steps

1. ✅ Understand this implementation completely
2. ⬜ Implement Ridge Regression (L2 regularization)
3. ⬜ Implement Lasso Regression (L1 regularization)
4. ⬜ Add polynomial features
5. ⬜ Move to Logistic Regression (classification)
6. ⬜ Build a neural network (multi-layer)

## Resources

- [Linear Regression Math](https://en.wikipedia.org/wiki/Linear_regression)
- [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)
- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo)
- [3Blue1Brown: Gradient Descent](https://www.youtube.com/watch?v=IHZwWFHWa-w)

---

**Key Insight**: This same training loop (forward pass → loss → gradients → update) is used in ALL modern deep learning. You've just learned the foundation of AI! 🚀
