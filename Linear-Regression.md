# Linear Regression

This document provides a comprehensive overview of the linear regression algorithm, including its motivation, mathematical formulation, and optimization techniques such as batch gradient descent, stochastic gradient descent, and the normal equations.

---

## Introduction

Linear regression is one of the simplest supervised learning algorithms, particularly for regression problems where the output is a continuous value. A practical example is used to motivate the concept, followed by definitions of key notations and concepts.

## Motivating Linear Regression

Linear regression maps input features to a continuous output, such as predicting house prices based on size and other attributes.

**Example: House Price Prediction**

* Dataset from Craigslist, Portland, Oregon
* Features: house size (square feet) and asking prices (thousands of dollars)
* Example data point: House of 2,104 sq ft with an asking price of $400,000
* Goal: Fit a straight line to the data with house size on x-axis and price on y-axis

## Supervised Learning Context

* A training set (house sizes and prices) is fed into a learning algorithm
* The algorithm outputs a **hypothesis function** mapping inputs to predicted outputs
* For a new house size, the hypothesis estimates the price

## Supervised Learning vs. Classification

* Linear regression: regression problem, output is continuous (price)
* Classification: discrete outputs (e.g., speed categories), covered in later lectures

---

# Key Design Choices in Learning Algorithms

* **Hypothesis:** How the model represents the relationship between inputs and outputs
* **Dataset:** Collection of training examples used to train the model
* **Optimization:** How model parameters are chosen to best fit the data

---

# Linear Regression: Hypothesis Representation

## Single Feature Case

Hypothesis: h(x) = θ₀ + θ₁ x

* θ₀: Intercept (bias term)
* θ₁: Slope (weight for the feature)
* x: Input feature (e.g., house size)
* h(x): Predicted price

## Multiple Feature Case

Hypothesis for multiple features (x₁ = size, x₂ = bedrooms):
h(x) = θ₀ + θ₁ x₁ + θ₂ x₂

Using a dummy feature x₀ = 1 for intercept:
h(x) = θ₀ x₀ + θ₁ x₁ + ... + θₙ xₙ

* x₀ = 1: Dummy feature for intercept
* x₁,...,xₙ: Input features
* θ₀,...,θₙ: Parameters
* n: Number of features

### Vector Notation

* Feature vector: x = [x₀, x₁, ..., xₙ]ᵀ
* Parameter vector: θ = [θ₀, θ₁, ..., θₙ]ᵀ
* Both vectors are (n+1)-dimensional

---

# Terminology and Notation

* Parameters: θ, tuned to make accurate predictions
* Training examples: m, number of rows in dataset
* Features: inputs x
* Target variable: y
* Training example: (x⁽ⁱ⁾, y⁽ⁱ⁾) for the i-th example
* Number of features: n (excluding dummy feature)

**Example:**

* x⁽¹⁾ = [1, 2104, 3]ᵀ
* y⁽¹⁾ = 400
* m = 49
* n = 2 (size and bedrooms)

---

# Linear Regression Algorithm

Goal: Choose θ such that hθ(x) predicts outputs close to y

## Cost Function

Squared error cost function:
J(θ) = (1/2) Σ (hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²

* Factor 1/2 simplifies derivatives
* Objective: Minimize J(θ)
* Squared error corresponds to Gaussian assumption

---

# Gradient Descent

Iterative optimization algorithm to minimize J(θ)

## Batch Gradient Descent

Algorithm:

1. Initialize θ = 0
2. Repeat until convergence:
   θⱼ := θⱼ - α Σ (hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾) xⱼ⁽ⁱ⁾ for j = 0,...,n

* α: Learning rate
* Visualization: J is a quadratic surface; gradient descent moves toward global minimum

### Learning Rate (α)

* Too large: overshoot minimum
* Too small: slow convergence
* Empirical tuning recommended

### Disadvantage

* Requires computing gradient over entire dataset, costly for large m

## Stochastic Gradient Descent (SGD)

Algorithm:

1. Initialize θ
2. For each training example i:
   θⱼ := θⱼ - α (hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾) xⱼ⁽ⁱ⁾ for j = 0,...,n

* Updates after each example, noisier path, faster for large datasets
* Convergence: oscillates near minimum; decreasing α reduces oscillations

### Practical Use

* Small datasets: batch gradient descent
* Large datasets: stochastic or mini-batch gradient descent

---

# Normal Equations

Closed-form solution for θ without iteration

## Matrix Notation

* Design matrix X = [x⁽¹⁾ᵀ; ...; x⁽ᵐ⁾ᵀ]
* Target vector y = [y⁽¹⁾, ..., y⁽ᵐ⁾]ᵀ
* Hypothesis: X θ = [hθ(x⁽¹⁾), ..., hθ(x⁽ᵐ⁾)]ᵀ
* Cost function: J(θ) = (1/2) (X θ - y)ᵀ (X θ - y)

## Derivation

* Expand J: J(θ) = (1/2) (θᵀ Xᵀ X θ - θᵀ Xᵀ y - yᵀ X θ + yᵀ y)
* Gradient: ∇J(θ) = Xᵀ X θ - Xᵀ y
* Set to zero: Xᵀ X θ = Xᵀ y
* Solve: θ = (Xᵀ X)⁻¹ Xᵀ y

### Handling Non-Invertible XᵀX

* Indicates redundant or linearly dependent features
* Use pseudo-inverse or remove redundant features

---

# Practical Considerations

* Batch vs. SGD: Batch for small datasets; SGD or mini-batch for large datasets
* Learning Rate Tuning: Empirical selection, monitor J(θ)
* Normal Equations vs. Gradient Descent: Normal equations efficient but less general; gradient descent more versatile

---

# Visualization

* Batch Gradient Descent: Smooth convergence to minimum
* SGD: Noisier path, faster for large datasets, oscillates near minimum

---

# Conclusion

Linear regression is a foundational supervised learning algorithm for regression problems. It involves:

* Defining a linear hypothesis
* Minimizing a squared error cost function
* Optimizing parameters using gradient descent or normal equations

Concepts like features, parameters, and cost function introduced here will be used throughout the course. Additional practice problems reinforce these concepts.
