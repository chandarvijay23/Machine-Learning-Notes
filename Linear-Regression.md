# Linear Regression

This document provides a comprehensive overview of the linear regression algorithm, including its motivation, mathematical formulation, and optimization techniques such as batch gradient descent, stochastic gradient descent, and the normal equations. 

## Introduction

The lecture introduces linear regression as one of the simplest supervised learning algorithms, particularly for regression problems where the output is a continuous value. The discussion begins with a practical example to motivate the concept and proceeds to define key notations and concepts that will be used throughout the course.

## Motivating Linear Regression

Linear regression is a fundamental algorithm in supervised learning, where the goal is to map input features to a continuous output. To illustrate, consider the example of predicting house prices based on their size and other attributes.

**Example: House Price Prediction**

- A dataset from Craigslist in Portland, Oregon, is used, containing the size of houses (in square feet) and their asking prices (in thousands of dollars).
- Example data point: A house of 2,104 square feet with an asking price of $400,000.
- When plotted, the dataset shows house size on the x-axis and price on the y-axis, and the goal is to fit a straight line to this data.

### Sample Housing Dataset

The following table shows a portion of our training dataset with 49 houses (m = 49):

| Example # | Size (sq ft) | Bedrooms | Price ($1000s) |
|-----------|-------------|----------|----------------|
| 1         | 2,104       | 3        | 400            |
| 2         | 1,600       | 3        | 330            |
| 3         | 2,400       | 3        | 369            |
| 4         | 1,416       | 2        | 232            |
| 5         | 3,000       | 4        | 540            |
| 6         | 1,985       | 4        | 300            |
| 7         | 1,534       | 3        | 315            |
| 8         | 1,427       | 3        | 199            |
| 9         | 1,380       | 3        | 212            |
| 10        | 1,494       | 3        | 243            |
| 11        | 1,940       | 4        | 347            |
| 12        | 2,000       | 3        | 330            |
| 13        | 1,890       | 3        | 310            |
| 14        | 4,478       | 5        | 700            |
| 15        | 1,268       | 3        | 200            |
| ...       | ...         | ...      | ...            |
| 49        | 1,552       | 3        | 255            |

**Dataset notation using this data:**
- **x**⁽¹⁾ = [1, 2104, 3]ᵀ (dummy feature, size, bedrooms for house #1)
- y⁽¹⁾ = 400 (price in thousands for house #1)
- **x**⁽²⁾ = [1, 1600, 3]ᵀ
- y⁽²⁾ = 330
- m = 49 (total number of training examples)
- n = 2 (number of features: size and bedrooms, excluding the dummy feature)

**Supervised Learning Context**

- In supervised learning, a training set (e.g., house sizes and prices) is fed into a learning algorithm.
- The algorithm outputs a function, termed a hypothesis, which maps inputs (e.g., house size) to predicted outputs (e.g., price).
- For a new house size, the hypothesis estimates the price.

**Supervised Learning vs. Classification**

- Linear regression is a regression problem because the output (price) is continuous.
- In contrast, classification problems involve discrete outputs (e.g., speed categories), which will be covered in a later lecture.

## Key Design Choices in Learning Algorithms

Designing a learning algorithm involves several critical decisions:

1. **Defining the Hypothesis**: How the model represents the relationship between inputs and outputs.
2. **Dataset**: The collection of training examples used to train the model.
3. **Optimization**: How the model parameters are chosen to best fit the data.

These choices are foundational for all supervised learning algorithms, and linear regression serves as a simple case to illustrate them.

## Linear Regression: Hypothesis Representation

In linear regression, the hypothesis is represented as a linear (or technically, affine) function of the input features.

**Single Feature Case**

For a single feature, such as house size ( x ), the hypothesis is:

$$h(x) = \theta_0 + \theta_1 x$$

where:
- θ₀: Intercept (bias term).
- θ₁: Slope (weight for the feature).
- x: Input feature (e.g., house size).
- The output h(x) is the predicted price.

**Multiple Feature Case**

For multiple features (e.g., house size x₁ and number of bedrooms x₂), the hypothesis generalizes to:

$$h(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2$$

To simplify notation, a dummy feature x₀ = 1 is introduced, allowing the hypothesis to be written as:

$$h(x) = \sum_{j=0}^n \theta_j x_j$$

where:
- x₀ = 1: Dummy feature for the intercept.
- x₁, x₂, ..., xₙ: Input features (e.g., size, number of bedrooms).
- θ₀, θ₁, θ₂, ..., θₙ: Parameters (weights) of the model.
- n: Number of features (e.g., n = 2 for size and bedrooms).

**Vector Notation**

- The feature vector is **x** = [x₀, x₁, x₂, ..., xₙ]ᵀ, where x₀ = 1.
- The parameter vector is **θ** = [θ₀, θ₁, θ₂, ..., θₙ]ᵀ.
- Both **x** and **θ** are (n+1)-dimensional vectors.

## Terminology and Notation

The lecture introduces standard notation used throughout the course:

- **Parameters**: **θ** represents the parameters of the learning algorithm, which are tuned to make accurate predictions.
- **Training Examples**: m denotes the number of training examples (e.g., rows in the house price dataset).
- **Features**: Inputs **x** are also called features or attributes.
- **Target Variable**: The output y is the target variable (e.g., house price).
- **Training Example**: A pair (**x**⁽ⁱ⁾, y⁽ⁱ⁾) represents the i-th training example, where:
  - **x**⁽ⁱ⁾: Feature vector for the i-th example.
  - y⁽ⁱ⁾: Target output for the i-th example.
  - The superscript (i) denotes the index of the training example (not exponentiation).

- **Number of Features**: n denotes the number of features (excluding the dummy feature x₀).

For example, in the house price dataset:

- **x**⁽¹⁾ = [1, 2104, 3]ᵀ (for a house with 2,104 square feet and 3 bedrooms).
- y⁽¹⁾ = 400 (price in thousands of dollars).
- m = 49 (number of training examples).
- n = 2 (features: size and number of bedrooms).

## Linear Regression Algorithm

The goal of linear regression is to choose parameters **θ** such that the hypothesis h_θ(**x**) predicts outputs close to the actual target values y.

### Cost Function

The cost function measures how well the hypothesis fits the training data. Linear regression, also known as ordinary least squares, uses the squared error cost function:

$$J(\boldsymbol{\theta}) = \frac{1}{2} \sum_{i=1}^m \left( h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) - y^{(i)} \right)^2$$

**Components:**

- h_θ(**x**⁽ⁱ⁾) = Σⱼ₌₀ⁿ θⱼxⱼ⁽ⁱ⁾: Predicted output for the i-th example.
- y⁽ⁱ⁾: Actual output for the i-th example.
- m: Number of training examples.
- The factor 1/2 is included for mathematical convenience (it simplifies derivatives).

**Objective**: Minimize J(**θ**) with respect to **θ**.

**Why Squared Error?**

- The squared error is used because it corresponds to a Gaussian assumption in generalized linear models (to be discussed in a later lecture).
- Alternatives like absolute error or higher powers (e.g., error to the fourth) are less common and will be justified later.

## Gradient Descent

Gradient descent is an iterative optimization algorithm used to minimize the cost function J(**θ**).

### Batch Gradient Descent

**Algorithm:**

1. Initialize **θ** (e.g., to a vector of zeros, **0**).
2. Repeat until convergence:

$$\theta_j := \theta_j - \alpha \sum_{i=1}^m \left( h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)} \quad \text{for } j = 0, 1, \ldots, n$$

where:
- α: Learning rate (controls step size).
- Σᵢ₌₁ᵐ (h_θ(**x**⁽ⁱ⁾) - y⁽ⁱ⁾) xⱼ⁽ⁱ⁾: Partial derivative of J(**θ**) with respect to θⱼ.
- := : Denotes assignment (not equality).

**Visualization:**

- Imagine J(**θ**) as a surface in (n+1)-dimensional space, with axes corresponding to θ₀, θ₁, ..., θₙ.
- The goal is to find the parameters **θ** that minimize the height of the surface (i.e., J(**θ**)).
- At each step, gradient descent takes a small step in the direction of the steepest descent (negative gradient).
- For linear regression, J(**θ**) is a quadratic function (a "bowl" shape) with a single global minimum, ensuring convergence to the optimal solution.

**Learning Rate (α):**

- Choosing α is empirical. Common starting values (e.g., 0.01) are adjusted based on performance.
- If α is too large, the algorithm may overshoot the minimum.
- If α is too small, convergence is slow.
- Practical approach: Try values on an exponential scale (e.g., 0.01, 0.02, 0.04, 0.08) and monitor J(**θ**).
- If J(**θ**) increases, reduce α.

**Disadvantage:**

- Batch gradient descent requires computing the gradient over the entire dataset (m examples) for each update.
- For large datasets (e.g., millions or billions of examples), this is computationally expensive, as it involves scanning the entire dataset for each step.

### Stochastic Gradient Descent

**Algorithm:**

1. Initialize **θ**.
2. For each training example i = 1, 2, ..., m, update:

$$\theta_j := \theta_j - \alpha \left( h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)} \quad \text{for } j = 0, 1, \ldots, n$$

Unlike batch gradient descent, stochastic gradient descent updates **θ** after processing each training example.

**Visualization:**

- The path to the minimum is noisier because updates are based on individual examples.
- On average, the algorithm moves toward the global minimum but may oscillate around it due to the randomness of example selection.
- For large datasets, stochastic gradient descent is faster because it updates parameters without scanning the entire dataset.

**Convergence:**

- Stochastic gradient descent does not converge to a fixed point but oscillates near the minimum.
- To reduce oscillations, the learning rate α can be decreased over time.
- Monitor J(**θ**) over time and stop when it stabilizes.

**Practical Use:**

- For small datasets (e.g., hundreds or thousands of examples), batch gradient descent is preferred due to its simplicity.
- For large datasets (e.g., millions of examples), stochastic gradient descent (or mini-batch gradient descent, which processes small batches of examples) is more efficient.

## Normal Equations

For linear regression, an alternative to gradient descent is the normal equations, which provide a closed-form solution to find the optimal **θ** in one step.

### Matrix Notation

**Design Matrix:**

Define the matrix **X** (called the design matrix) with m rows (training examples) and n+1 columns (features, including x₀ = 1):

<img src="https://latex.codecogs.com/svg.image?\mathbf{X}=\begin{bmatrix}1&2104&3\\1&1600&3\\1&2400&3\\1&1416&2\\\vdots&\vdots&\vdots\\1&1552&3\end{bmatrix}" />

Each row is the feature vector **x**⁽ⁱ⁾ = [1, x₁⁽ⁱ⁾, x₂⁽ⁱ⁾]ᵀ.

**Target Vector:**

Define the vector **y** containing the target values:

<img src="https://latex.codecogs.com/svg.image?\mathbf{y}=\begin{bmatrix}400\\330\\369\\232\\\vdots\\255\end{bmatrix}" />

**Hypothesis in Matrix Form:**

The predictions are:

<img src="https://latex.codecogs.com/svg.image?\mathbf{X}\boldsymbol{\theta}=\begin{bmatrix}h_{\boldsymbol{\theta}}(\mathbf{x}^{(1)})\\h_{\boldsymbol{\theta}}(\mathbf{x}^{(2)})\\\vdots\\h_{\boldsymbol{\theta}}(\mathbf{x}^{(m)})\end{bmatrix}" />

**Cost Function in Matrix Form:**

The cost function can be written as:

$$J(\boldsymbol{\theta}) = \frac{1}{2} (\mathbf{X} \boldsymbol{\theta} - \mathbf{y})^T (\mathbf{X} \boldsymbol{\theta} - \mathbf{y})$$

This represents the sum of squared errors over all training examples.

### Derivation of Normal Equations

**Objective**: Minimize J(**θ**) by taking the derivative with respect to **θ**, setting it to zero, and solving for **θ**.

**Matrix Derivatives:**

The derivative of J(**θ**) with respect to **θ** is computed using matrix calculus.

Define the gradient as a vector:

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \begin{bmatrix}\frac{\partial J}{\partial \theta_0} \\\frac{\partial J}{\partial \theta_1} \\\vdots \\\frac{\partial J}{\partial \theta_n}\end{bmatrix}$$

**Useful properties:**
- Trace of a matrix **A**: trace(**A**) = Σᵢ Aᵢᵢ.
- trace(**A**) = trace(**A**ᵀ).
- trace(**AB**) = trace(**BA**).
- trace(**ABC**) = trace(**CAB**) (cyclic permutation).
- For a function f(**A**) = trace(**AB**), the derivative is:
  $$\frac{\partial f}{\partial \mathbf{A}} = \mathbf{B}^T$$
- For f(**A**) = trace(**A** **A**ᵀ **C**), the derivative is:
  $$\frac{\partial f}{\partial \mathbf{A}} = 2 \mathbf{C} \mathbf{A}$$

**Steps:**

1. Expand J(**θ**):
   $J(\boldsymbol{\theta}) = \frac{1}{2} (\boldsymbol{\theta}^T \mathbf{X}^T \mathbf{X} \boldsymbol{\theta} - \boldsymbol{\theta}^T \mathbf{X}^T \mathbf{y} - \mathbf{y}^T \mathbf{X} \boldsymbol{\theta} + \mathbf{y}^T \mathbf{y})$

2. Take the derivative:
   $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbf{X}^T \mathbf{X} \boldsymbol{\theta} - \mathbf{X}^T \mathbf{y}$

3. Set the derivative to zero:
   $\mathbf{X}^T \mathbf{X} \boldsymbol{\theta} = \mathbf{X}^T \mathbf{y}$

4. Solve for **θ**:
   $\boldsymbol{\theta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$

**Normal Equations:**

The solution is:

$$\boldsymbol{\theta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

This formula computes the optimal **θ** in one step, without iteration.

**Handling Non-Invertible **X**ᵀ**X**:**

- If **X**ᵀ**X** is non-invertible, it typically indicates redundant or linearly dependent features.
- Use the pseudo-inverse to compute a solution, or remove redundant features to avoid this issue.

## Practical Considerations

**Batch vs. Stochastic Gradient Descent:**

- Use batch gradient descent for small datasets where computation is manageable.
- Use stochastic gradient descent (or mini-batch gradient descent) for large datasets to improve efficiency.
- Mini-batch gradient descent, which processes small batches of examples, is a common compromise (discussed in other courses like CS230).

**Learning Rate Tuning:**

- The learning rate α is chosen empirically by testing multiple values.
- Monitor J(**θ**) to ensure it decreases; adjust α if it increases.
- For stochastic gradient descent, gradually decreasing α can reduce oscillations.

**Normal Equations vs. Gradient Descent:**

- The normal equations are efficient for linear regression but do not generalize to other algorithms (e.g., neural networks or generalized linear models).
- Gradient descent is more versatile and applicable to a wide range of machine learning algorithms.

## Visualization of Gradient Descent

**Batch Gradient Descent:**

- Starts with an initial hypothesis (e.g., a horizontal line if **θ** = **0**).
- Each iteration updates **θ**, adjusting the line to better fit the data.
- Example: After several iterations, the line converges to a good fit for the house price dataset.

**Stochastic Gradient Descent:**

- Updates **θ** after each example, leading to a noisier path but faster progress for large datasets.
- The algorithm may not converge exactly but oscillates near the minimum.

## Conclusion

Linear regression is a foundational supervised learning algorithm for regression problems. It involves defining a linear hypothesis, minimizing a squared error cost function, and optimizing parameters using either gradient descent (batch or stochastic) or the normal equations. The concepts and notation introduced here (e.g., features, parameters, cost function) will be used throughout the course for more complex algorithms.
