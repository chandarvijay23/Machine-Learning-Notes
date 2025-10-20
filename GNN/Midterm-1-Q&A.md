# Midterm-1 Q&A Graph Neural Networks

## Question 1 (2 points). Define Statistical Risk Minimization and give at least two examples of different losses.

**Statistical Risk Minimization (SRM)** is the optimization program used to formulate learning as a mathematical process. It involves averaging the loss $\mathcal{L}(y, \Phi(x))$ over **nature's probability distribution $p(x, y)$** and choosing the best estimator or classifier $\Phi$.

The formulation seeks the optimal function $\Phi^*$ that minimizes this average cost (risk) over all possible estimators:
$$\Phi^* = \operatorname{argmin}_\Phi E_{p(x,y)} \left[ \mathcal{L}(y, \Phi(x)) \right]$$.

The AI predicts an output $\hat{y} = \Phi(x)$, while nature draws the actual output $y$ according to $p(x, y)$; the SRM then minimizes the expectation of the resulting loss. The outcome of solving this problem (learning/training) is the function $\Phi^*$ with the minimum average statistical loss.

Two examples of loss functions $\mathcal{L}(y, \hat{y})$ that measure the cost of predicting $\hat{y}$ when the actual output is $y$ are:

1.  **Quadratic loss:** Often used in estimation problems, it measures the squared Euclidean distance between the actual output and the prediction:
    $$\mathcal{L}(y, \hat{y}) = \|y - \hat{y}\|_2^2$$.
2.  **Hit loss:** Often used in classification problems, it counts the number of components where the real output $y$ and its prediction $\hat{y}$ disagree:
    $$\mathcal{L}(y, \hat{y}) = \|y - \hat{y}\|_0 = \text{number of components where } y \neq \hat{y}$$.

## Question 2 (2 points). Explain why Empirical Risk Minimization without the use of a learning parameterization is a nonsensical formulation of machine learning.

**Empirical Risk Minimization (ERM)** replaces the statistical risk (an expectation over the unknown distribution $p(x, y)$) with the **empirical risk** (an average over a finite training dataset $\mathcal{T}$).

The unconstrained ERM problem is formulated as:
$$\Phi^*_E = \operatorname{argmin}_\Phi \frac{1}{Q}\sum_{q=1}^Q \mathcal{L}(y_q, \Phi(x_q))$$.

This formulation is **nonsensical** and **trivial** because the minimization can be trivially solved. The optimal solution $\Phi^*_E$ is achieved simply by having the AI **copy the true output $y_q$ for every input $x_q$ that appears in the training set** (i.e., making $\Phi(x_q) = y_q$). This makes all pointwise losses vanish, resulting in a minimum possible empirical risk of null.

However, this solution is nonsensical because **it yields no information whatsoever about observations that are outside the training set**. The behavior of the function $\Phi$ for inputs $x$ not in $\mathcal{T}$ can be entirely arbitrary, meaning the AI has failed to generalize or "learned nothing". To solve this, a restriction on the search space, known as a learning parameterization (function class $\mathcal{C}$), is required.

## Question 3 (2 points). Define Empirical Risk Minimization with learning parameterizations and exemplify with a linear parameterization and a neural network parameterization.

**Empirical Risk Minimization (ERM) with learning parameterizations** is a sensible formulation of machine learning achieved by introducing a **function class $\mathcal{C}$**. Instead of searching for an optimal function over the space of all possible functions (which is nonsensical), the search is restricted to those functions belonging to $\mathcal{C}$.

The parametrized ERM problem is defined as:
$$\Phi^* = \operatorname{argmin}_{\Phi \in \mathcal{C}} \frac{1}{Q}\sum_{q=1}^Q \mathcal{L}(y_q, \Phi(x_q))$$.

If the function class $\mathcal{C}$ is sufficiently smooth and the training set size $Q$ is sufficiently large, the optimal ERM solution $\Phi^*$ becomes a valid approximation of the optimal Statistical Risk Minimization solution $\Phi^*_S$.

#### Examples of Parameterizations:

1.  **Linear Parameterization:**
    If the function class $\mathcal{C}$ is chosen to be the set of linear functions, the AI map is restricted to the form $\Phi(x) = Hx$ for a matrix parameter $H$. The optimization problem seeks the optimal matrix $H^*$:
    $$H^* = \operatorname{argmin}_H \frac{1}{Q}\sum_{q=1}^Q \mathcal{L}(y_q, Hx_q)$$.

2.  **Neural Network Parameterization (Graph Neural Network):**
    A neural network, such as a Graph Neural Network (GNN), composes a cascade of layers, each consisting of a linear map restricted to a convolution (a graph filter) followed by a pointwise nonlinearity $\sigma$. The function class $\mathcal{C}$ is spanned by the **filter tensor $\mathcal{H}$** (which groups all filter coefficients $h_{\ell k}$ across layers) and the graph shift operator $S$. The output is represented as $x_L = \Phi(x; S, \mathcal{H})$.

## Question 4 (2 points). What are the three fundamental components of an AI system? Of these three components there is one whose choice is most important and most under the control of the system's designer. Which is it? What is the property that this choice controls?

The three fundamental components of an Artificial Intelligence (AI) or Machine Learning (ML) system, formulated as an Empirical Risk Minimization (ERM) problem, are:

1.  **A dataset ($\mathcal{T}$):** Containing input-output pairs $(x, y)$ that describe the relationship to be mimicked.
2.  **A pointwise loss function ($\mathcal{L}$):** Used to evaluate the fit of predictions $\Phi(x)$ relative to the actual output $y$.
3.  **A function class ($\mathcal{C}$) (or learning parametrization):** Which restricts the search space of possible AI maps.

**Most Important Component and Designer Control:**

The **function class $\mathcal{C}$ (or learning parametrization)** is the component whose choice is **most important** and **most under the control of the system's designer**. The data must be acquired, and the loss function is usually a given metric, leaving the function class as the primary degree of freedom available.

**Property Controlled:**

The function class determines the properties of the AI because it controls **generalization**. Generalization refers to how the AI performs estimates for inputs $x$ that are *not* part of the training set. If the parametrization is not matched to the underlying system's relationship, the learning process will fail to generalize.

## Question 5 (2 points). Define gradient descent and stochastic gradient descent. Explain their differences.

#### Gradient Descent (GD) Definition
Gradient Descent is an algorithm used to minimize the average loss function $L(H)$ parameterized by $H$. It utilizes the gradient $g(H) = \nabla L(H)$, which is perpendicular to the level sets of the loss and points towards the minimum. The update rule is:
$$H^{t+1} = H^t - \epsilon g(H^t)$$.
The gradient $g(H)$ is calculated by taking the average of the gradients of all $Q$ pointwise losses across the entire training dataset $\mathcal{T}$:
$$g(H) = \frac{1}{Q}\sum_{q=1}^Q \nabla_H \mathcal{L}(y_q, \Phi(x_q; H))$$.

#### Stochastic Gradient Descent (SGD) Definition
Stochastic Gradient Descent is an algorithm that addresses the high computational cost of GD by **replacing the full gradient $g(H)$ with a stochastic gradient $\hat{g}(H)$**. At each iteration $t$, SGD selects a small batch $T_t$ of $Q_t \ll Q$ samples randomly from the dataset $\mathcal{T}$. The stochastic gradient is defined as the average over this small batch:
$$\hat{g}(H^t) = \frac{1}{Q_t} \sum_{(x_q, y_q) \in T_t} \nabla_H \mathcal{L}(y_q, \Phi(x_q; H^t))$$.
The update rule is:
$$H^{t+1} = H^t - \epsilon \hat{g}(H^t)$$.

#### Differences
| Feature | Gradient Descent (GD) | Stochastic Gradient Descent (SGD) |
| :--- | :--- | :--- |
| **Gradient Calculation** | Uses all $Q$ samples to compute the full gradient $g(H)$. | Uses a small random batch $Q_t \ll Q$ samples to compute $\hat{g}(H)$. |
| **Computational Cost** | High cost, as it computes $Q$ pointwise gradients per step. | Much cheaper, as it sums over a smaller number of pointwise gradients. |
| **Direction** | Negative gradient $-g(H)$ reliably points toward the optimum. | Stochastic gradient $\hat{g}(H)$ points toward the optimum **on expectation**. |

The difference in computational cost between GD and SGD is crucial in practice; the cheaper calculation of $\hat{g}(H)$ is often "the difference between a method that works and a method that does not work".

## Question 6 (2 points). Convergence of stochastic gradient descent holds for limit infimums. Explain how this is different from regular convergence and why does it matter in practice.

**SGD Convergence Definition:**
The convergence of Stochastic Gradient Descent (SGD) holds for **limit infimums**. This is expressed as a bound on the distance between the iterate $H^t$ and the optimum $H^*$:
$$\liminf_{t \to \infty} \|H^t - H^*\|^2 \leq O\left( \frac{\epsilon}{\sqrt{Q_t}} \right)$$.

**Difference from Regular Convergence:**
Regular convergence, $\lim_{t \to \infty} \|H^t - H^*\| = 0$, implies that the iterates $H^t$ approach the optimal value $H^*$ exactly as time $t$ tends to infinity. However, because SGD only converges in the limit infimum sense, it means that the iterates $H^t$ **approach the optimum $H^*$ and then hover around it**, rather than converging exactly.

**Why it Matters in Practice:**
In practice, this convergence behavior results in a **hover region** around the optimum $H^*$. The size of this hover region is controlled by the algorithmic parameters:

1.  **Stepsize ($\epsilon$):** The size of the hover region is **proportional** to the stepsize $\epsilon$. To get closer to the optimum (decrease the hover region), the stepsize must be decreased, which results in slower convergence (more iterations needed).
2.  **Batch Size ($Q_t$):** The size of the hover region is **inversely proportional** to the square root of the batch size $\sqrt{Q_t}$. Increasing the batch size decreases the hover region, but this requires a larger computational cost per stochastic gradient calculation.

## Question 7 (2 points). If the learning parametrization is not matched to the underlying system's input-to-output relationship we do not expect learning to work. Explain.

The learning parametrization, or function class $\mathcal{C}$, acts as a hypothesis or a **model** of how outputs are related to inputs, and it must be an **accurate representation of nature**. If this parametrization is not matched to the underlying system's input-to-output relationship (the actual model), we do not expect learning to work well.

The sources illustrate this with a case where the **model is mismatched**:

1.  **Scenario:** Data is generated by a highly non-linear *sign model*. The designer uses a simple *linear parametrization* ($\Phi(x)=Hx$).
2.  **Result:** Although the optimization (SGD) succeeds in solving the Empirical Risk Minimization problem (the loss is reduced on the training set), it converges to a high loss value. It is concluded that the system is *not learning* because the resultant AI is not good.
3.  **Generalization Failure:** When tested outside the training set, the performance is "just as bad". The linear parametrization failed to learn the non-linear relationship.

Therefore, learning fails when the parametrization is mismatched because the chosen function class $\mathcal{C}$ is fundamentally incapable of representing the actual relationship between inputs and outputs.

## Question 8 (2 points). If the learning parametrization is matched to the underlying system's input-to-output relationship we expect learning to work but only if we have sufficient data relative to the complexity of the problem. Explain

Even when the learning parametrization is **matched** to the underlying input-to-output relationship (e.g., using a linear parametrization for data generated by a linear model), successful learning still requires **sufficient data** relative to the problem's complexity.

The sources illustrate a scenario where learning fails due to insufficient data despite the match:

1.  **Scenario:** A *linear model* is used to generate data, and a *linear parametrization* is used for learning (matched relationship). However, the number of data samples ($Q$) is reduced significantly relative to the problem dimensions.
2.  **Training Result:** The SGD algorithm successfully reduces the loss to a small value on the training set, indicating the AI learned to predict outputs within the training set.
3.  **Generalization Failure:** When operating outside the training set (live operation), the loss is **not reduced by much**. This failure occurs because there was **not enough data** to learn the model and generalize effectively.

This leads to the observation that designers must choose models that are not only matched but also of **sufficiently limited complexity** (e.g., leveraging structure using convolutional architectures like CNNs and GNNs) so that they can learn effectively even with data amounts that are typically "always insufficient" when problem dimensions are large.

## Question 9 (2 points). Define the Adjacency and Laplacian matrix of a graph along with their normalized versions. The eigenvalues of the Laplacian are nonnegative. One of them is zero. Prove and explain.

**Definitions:**

*   **Adjacency Matrix ($A$):** A sparse matrix where the $i, j$ entry $A_{ij}$ is non-zero if and only if the pair $(i, j)$ is an edge of the graph. If non-zero, $A_{ij}$ records the **weight $w_{ij}$**.
*   **Laplacian Matrix ($L$):** Defined as the difference between the **degree matrix $D$** (a diagonal matrix containing the node degrees $d_i$ on the diagonal) and the adjacency matrix $A$:
    $$L = D - A = \text{diag}(A\mathbf{1}) - A$$.
*   **Normalized Adjacency Matrix ($\bar{A}$):** Defined using the inverse square root of the degree matrix $D$:
    $$\bar{A} := D^{-1/2} A D^{-1/2}$$. The entries are $(\bar{A})_{ij} = w_{ij} / \sqrt{d_i d_j}$.
*   **Normalized Laplacian Matrix ($\bar{L}$):** Defined using the same normalization:
    $$\bar{L} := D^{-1/2} L D^{-1/2}$$. It can also be written as the difference between the identity matrix ($I$) and the normalized adjacency matrix ($\bar{A}$): $\bar{L} = I - \bar{A}$.

**Eigenvalue Property (One is Zero):**

The eigenvalues of the Laplacian matrix $L$ are nonnegative, and **one of them is zero**.

**Proof/Explanation for the Zero Eigenvalue:**
The sources implicitly support that the all-ones vector $\mathbf{1}$ is an eigenvector of $L$ with an eigenvalue of $0$.

1.  The **degree $d_i$ of node $i$** is the sum of the weights of its incident edges: $d_i = \sum_j w_{ij}$.
2.  The $i$-th entry of the product $A\mathbf{1}$ is $\sum_j A_{ij} (1) = \sum_j w_{ij}$, which is exactly the degree $d_i$. Thus, the vector $A\mathbf{1}$ is equal to the vector $\mathbf{d}$ containing all node degrees.
3.  The **degree matrix $D$** is a diagonal matrix where $D_{ii} = d_i$. Therefore, $D\mathbf{1}$ is also the vector $\mathbf{d}$.
4.  The Laplacian matrix is $L = D - A$. Applying the Laplacian matrix to the all-ones vector $\mathbf{1}$ yields:
    $$L\mathbf{1} = (D - A)\mathbf{1} = D\mathbf{1} - A\mathbf{1} = \mathbf{d} - \mathbf{d} = \mathbf{0}$$
5.  Since $L\mathbf{1} = 0 \cdot \mathbf{1}$, the all-ones vector $\mathbf{1}$ is an eigenvector of $L$ corresponding to the eigenvalue $\lambda = 0$.

## Question 10 (2 points). Define a graph convolutional filter of order k. Give an equation and draw a diagram.

**Definition:**
A graph convolutional filter is a specific type of filter used for the linear processing of graph signals. It is defined as a **polynomial (or series) on the graph shift operator $S$** with a given set of coefficients $h_k$.

Assuming the filter order is $K-1$ (meaning $K$ total taps, $k=0$ to $K-1$), the graph filter $H(S)$ is:
$$H(S) = \sum_{k=0}^{K-1} h_k S^k$$.

The resulting output signal $y$ when applied to an input signal $x$ is called the **graph convolution** of the filter $h$ with the signal $x$:
$$y = H(S) x = \sum_{k=0}^{K-1} h_k S^k x$$.

**Diagram (Graph Shift Register):**

The graph convolution is equivalent to a weighted linear combination of the **diffusion sequence** elements. It can be conceptualized and implemented using a **shift register** structure that performs three recursive operations: **Shift. Scale. Sum**.

The diagram illustrates the aggregation of shifted (diffused) versions of the input $x$:

| Stage | Operation |
| :---: | :---: |
| **Diffusion Sequence** | $\mathbf{S^0 x}$ (Input $x$) $\xrightarrow{S} \mathbf{S^1 x} \xrightarrow{S} \mathbf{S^2 x} \xrightarrow{S} \mathbf{S^3 x} \dots$ |
| **Scaling** | Scale each diffused signal by the coefficients $h_0, h_1, h_2, h_3, \dots$ |
| **Summation** | Sum the resulting terms $h_0 S^0 x + h_1 S^1 x + h_2 S^2 x + h_3 S^3 x + \dots$ to produce the output $y = h \star_S x$ |

$$\begin{array}{c} \mathbf{S^0 x} \xrightarrow{h_0} \\ \mathbf{S^1 x} \xrightarrow{h_1} \end{array} \quad \mathbf{y = \sum_{k=0}^{K-1} h_k S^k x} \quad \begin{array}{c} \mathbf{S^2 x} \xrightarrow{h_2} \\ \mathbf{S^3 x} \xrightarrow{h_3} \end{array} \quad \mathbf{}$$

## Question 11 (2 points). Define the diffusion sequence. Introduce its power series version and its recursive definition. Use the diffusion sequence to define a graph filter.

The **diffusion sequence** is produced by composing the graph shift operator $S$ with itself,. The $k$-th element of this sequence $x^{(k)}$ diffuses information to and from **$k$-hop neighborhoods**. This embedding of the trade-off between local and global information is one reason why the diffusion sequence is used to define graph filters,.

The diffusion sequence can be defined in two equivalent ways:

1.  **Recursive Definition:** The sequence begins with $x^{(0)}$, the graph signal itself, and subsequent elements are found by repeatedly multiplying by the shift operator $S$,,:
    $$x^{(k+1)} = S x^{(k)}, \quad \text{with } x^{(0)} = x$$
    (The sources strongly warn to **always use this recursive version in implementations** due to dramatic differences in computational cost.)
2.  **Power Series Version (Unrolled Recursion):** The $k$-th element of the sequence is written as the $k$-th power of the graph shift operator $S$ applied to the input signal $x$,:
    $$x^{(k)} = S^k x$$

**Defining a Graph Filter using the Diffusion Sequence:**

A **graph filter** $H(S)$ is defined as a polynomial or series on the graph shift operator $S$ with a given set of coefficients $h_k$,. Applying this filter to a signal $x$ results in the output signal $y$, which is equivalent to a **weighted linear combination of the elements of the diffusion sequence**,,:
$$y = H(S) x = \sum_{k=0}^{\infty} h_k S^k x$$

## Question 12 (2 points). A graph convolutional filter is a linear combination of the elements of the diffusion sequence. Explain with a diagram and write down the corresponding equation.

A graph convolutional filter results in an output signal $y$ that is a **weighted linear combination of the elements of the diffusion sequence**,,. Each element of the diffusion sequence, $S^k x$, aggregates information from $k$-hop neighborhoods, ensuring the output aggregates local and increasing non-local information.

**Equation:**
For a finite filter order $K-1$, the output $y$ is defined by combining the scaled diffusion elements:
$$y = h \star_S x = \sum_{k=0}^{K-1} h_k S^k x$$

**Diagram Explanation:**
This relationship is implemented using a "Shift. Scale. Sum" process, visualized by a shift register,.

1.  **Shift:** The input $x$ is repeatedly multiplied by the graph shift operator $S$ to generate the diffusion sequence elements ($S^0 x, S^1 x, S^2 x, \dots$),.
2.  **Scale:** Each diffusion element $S^k x$ is scaled by its corresponding filter coefficient $h_k$,.
3.  **Sum:** These scaled terms are then accumulated to produce the output $y$,.

$$\text{Diffusion Sequence: } S^0 x \xrightarrow{S} S^1 x \xrightarrow{S} S^2 x \xrightarrow{S} S^3 x \dots$$
$$\text{Output: } y = (h_0 S^0 x) + (h_1 S^1 x) + (h_2 S^2 x) + (h_3 S^3 x) + \dots$$
(This structure is typically visualized using interconnected summing junctions and scaling blocks, as detailed in the sources,).

## Question 13 (2 points). Explain a graph filter using a graph shift register. Explain a time filter using a time shift register. Discuss similarities and differences.

**Graph Filter (GF) using a Graph Shift Register:**
A graph convolution, $y = h \star_S x$, is computed efficiently using a **shift register structure** that follows the recursive definition of the diffusion sequence,. This process interprets the convolution as a combination of shifting, scaling, and summing,:
1.  **Shift:** The current signal $x^{(k)}$ is multiplied by the Graph Shift Operator $S$ to produce the next diffused signal $x^{(k+1)}$.
2.  **Scale:** The current signal $x^{(k)}$ is scaled by the coefficient $h_k$.
3.  **Sum:** The scaled result is added to the accumulating output signal $y$,.

**Time Filter (TF) using a Time Shift Register:**
A time convolution is conventionally a linear combination of time-shifted versions of the input signal, $y_n = \sum h_k x_{n-k}$,,. The shift register implementation involves taking the input signal $x$, using the **time shift operator** to generate shifted versions (like $x_{n-1}, x_{n-2}$, etc.), scaling them by coefficients $h_k$, and summing the results.

**Similarities and Differences:**

| Feature | Graph Filter (GF) | Time Filter (TF) |
| :--- | :--- | :--- |
| **Fundamental Structure** | Polynomial on shift operator $S$: $H(S) = \sum h_k S^k$. | Polynomial on shift operator $S$: $H(S) = \sum h_k S^k$. |
| **Shift Mechanism ($S$ definition)** | **Arbitrary Graph Shift Operator** (e.g., Adjacency Matrix of a general graph). | **Adjacency Matrix of a directed line graph**,. |
| **Implementation** | Both use the **identical Shift Register structure** (Shift. Scale. Sum) for implementation,. |
| **Generalization** | TF is a **particular case** of a GF, recovered when $S$ describes a line graph,,. |

The structure is the same, but the **underlying shift operator** ($S$) is what differentiates them, allowing the Graph Filter to generalize the convolution concept to arbitrary topologies.

## Question 14 (2 points). Implement a Pytorch class that creates filters of order K and given coefficients. It suffices to give the forward method that takes the graph signal x as an input. You can assume that S, K and coefficients hk have been initialized. You must use the diffusion sequence. Why?

Since this is a conceptual exercise based on the source material, the implementation logic focuses on the prescribed algebraic steps:

```python
# Assuming Python/PyTorch environment and initialization of:
# S: The Graph Shift Operator (e.g., torch.Tensor or specialized operator)
# K: The filter order (integer)
# hk: List or array of filter coefficients [h0, h1, ..., hK-1]

# Pytorch class equivalent:
class GraphFilter(nn.Module):
    def __init__(self, S, K, hk):
        super().__init__()
        self.S = S
        self.K = K
        # hk needs to be wrapped as a parameter if it were trainable,
        # but here we assume it's a fixed input set for calculation.
        self.hk = hk

    def forward(self, x):
        # x is the input graph signal (x^(0))
        y = 0  # Initialize output signal y (Accumulated sum)
        
        # Current element of the diffusion sequence, starts as x^(0) = S^0 * x
        x_k = x
        
        # Iterate k from 0 to K-1 (K total taps)
        for k in range(self.K):
            # 1. Scale: Scale the current diffused signal x_k by coefficient hk
            y += self.hk[k] * x_k  # Sum += h_k * x^(k)
            
            # 2. Shift: Calculate the next element of the diffusion sequence recursively
            # This implements x^(k+1) = S * x^(k)
            if k < self.K - 1:
                x_k = torch.matmul(self.S, x_k)
                
        # 3. Sum: The final result is y
        return y
```

**Why you must use the diffusion sequence (recursively):**

You must use the **recursive definition of the diffusion sequence** ($x^{(k+1)} = S x^{(k)}$) because of the **dramatic differences in computational cost** compared to the power version ($x^{(k)} = S^k x$). The computation of high powers of the shift operator $S^k$ is generally computationally demanding, whereas iteratively applying $S$ leads to the highly efficient shift register implementation,.

## Question 15 (2 points). Define the graph Fourier transform (GFT) and write down a graph filter in the graph frequency domain. Show a proof of your result.

**Definition of the Graph Fourier Transform (GFT):**
The analysis begins by assuming a symmetric graph shift operator $S$ with eigenvector decomposition $S = V \Lambda V^H$, where $V$ is the eigenvector matrix,. The **Graph Fourier Transform (GFT)** of a graph signal $x$ is defined as the projection of $x$ onto the eigenspace of $S$:
$$\tilde{x} = V^H x$$
The resulting vector $\tilde{x}$ is the representation of $x$ in the graph frequency domain.

**Graph Filter in the Graph Frequency Domain (Theorem):**
The GFTs of the input ($\tilde{x}$) and the filtered output ($y = H(S)x$, with $\tilde{y} = V^H y$) are related by:
$$\tilde{y} = \sum_{k=0}^{\infty} h_k \Lambda^k \tilde{x}$$,
This theorem shows that the filter operation, which is a polynomial on $S$ in the graph domain, becomes a polynomial on the eigenvalue matrix $\Lambda$ in the frequency domain.

**Proof of the Result:**
The proof leverages the spectral decomposition of $S$:

1.  Start with the definition of the filtered signal $y$ as a polynomial on $S$:
    $$y = \sum_{k=0}^{\infty} h_k S^k x$$
2.  Substitute the spectral decomposition $S^k = V \Lambda^k V^H$ into the expression:
    $$y = \sum_{k=0}^{\infty} h_k V \Lambda^k V^H x$$,
3.  To shift to the GFT domain, multiply both sides by $V^H$ from the left:
    $$V^H y = V^H \sum_{k=0}^{\infty} h_k V \Lambda^k V^H x$$,
4.  Since $V^H V = I$ (the identity matrix), the terms $V^H V$ cancel out within the sum,.
5.  Identify $\tilde{y} = V^H y$ and $\tilde{x} = V^H x$:
    $$\tilde{y} = \sum_{k=0}^{\infty} h_k \Lambda^k \tilde{x}$$,

## Question 16 (2 points). Define the frequency response of a graph filter. Use it to write down the input-output relationship of a graph filter in the GFT domain.

**Definition of Frequency Response ($h̃(\lambda)$):**
The frequency response of a graph filter, defined by coefficients $h = \{h_k\}_{k=0}^{\infty}$, is the **polynomial on a scalar variable $\lambda$** that uses the same filter coefficients,:
$$h̃(\lambda) = \sum_{k=0}^{\infty} h_k \lambda^k$$
The frequency response is defined by the coefficients $h_k$ and is therefore **independent of the specific graph**,.

**Input-Output Relationship in the GFT Domain:**
Because the matrix $\Lambda$ is diagonal, the operation of the graph filter in the frequency domain is equivalent to a **pointwise multiplication** (a diagonal matrix relationship),.

The full relationship is $\tilde{y} = \sum_{k=0}^{\infty} h_k \Lambda^k \tilde{x}$. Since $\Lambda^k$ is diagonal with entries $\lambda_i^k$, the $i$-th component of the output GFT ($\tilde{y}_i$) is related to the $i$-th component of the input GFT ($\tilde{x}_i$) by the frequency response evaluated at the corresponding eigenvalue ($\lambda_i$),,:
$$\tilde{y}_i = h̃(\lambda_i) \tilde{x}_i$$,

## Question 17 (2 points). What is the role of a graph in the instantiation of a filter whose frequency response if h(λ). Illustrate.

The graph plays a critical role in how a filter's theoretical frequency response $h̃(\lambda)$ is applied in practice.

**Role of the Graph:**
The frequency response $h̃(\lambda)$ is a polynomial determined solely by the filter coefficients $h_k$ and is **independent of the graph**,. The role of the graph is to **determine the specific eigenvalues ($\lambda_i$) on which the response is instantiated**,,.

**Illustration (Conceptual):**
The frequency response $h̃(\lambda)$ is a continuous analytic function defined over the scalar variable $\lambda$.

1.  **Without a Graph:** The function $h̃(\lambda)$ exists purely as a continuous polynomial dictated by $h_k$,.
2.  **With Graph S:** The output of the filter depends only on the values of $h̃(\lambda)$ evaluated at the discrete eigenvalues $\lambda_i$ determined by the shift operator $S$.
3.  **With Graph $\hat{S}$:** If a different graph $\hat{S}$ is used, the filter applies the exact same function $h̃(\lambda)$, but evaluates it at the different set of eigenvalues $\hat{\lambda}_i$ defined by $\hat{S}$,.

Thus, the graph provides the **discrete frequencies** where the continuous frequency response function is evaluated.

## Question 18 (2 points). Define a pointwise nonlinearity and a graph perceptron. Use these definition to give a recursive definition of a GNN with L layers.

**Pointwise Nonlinearity ($\sigma$):**
A pointwise nonlinearity is a nonlinear function applied to a vector $\mathbf{x}$ **componentwise, without mixing entries**,,. If $x$ is a vector with entries $x_1, \dots, x_n$, applying the nonlinearity $\sigma$ results in a vector where the $i$-th component is $\sigma(x_i)$,. Examples include the Rectified Linear Unit (ReLU) or the hyperbolic tangent,. Pointwise nonlinearities often reduce variability, functioning as **demodulators**,,.

**Graph Perceptron:**
A graph perceptron is a function class that aims to introduce nonlinearity beyond that of a simple graph filter,. It is defined as the composition of a **graph filter** with a **pointwise nonlinearity** $\sigma$,,. For filter coefficients $h_k$ and shift operator $S$, the graph perceptron is defined as:
$$\Phi(x) = \sigma \left[ \sum_{k=0}^{K-1} h_k S^k x \right]$$

**Recursive Definition of a GNN with L Layers:**
A Graph Neural Network (GNN) is defined by composing a cascade of $L$ graph perceptrons (layering perceptrons),,.

Let $x$ be the input signal, conventionally denoted as the Layer 0 output, $x^{(0)} = x$,. A generic Layer $\ell$ processes the output $x^{(\ell-1)}$ of the previous layer $(\ell-1)$,.

The **recursive definition** for the GNN output $x^{(\ell)}$ at Layer $\ell$ ($\ell = 1, \dots, L$) is:
$$x^{(\ell)} = \sigma \left[ z^{(\ell)} \right] = \sigma \left[ \sum_{k=0}^{K-1} h_{\ell, k} S^k x^{(\ell-1)} \right]$$
where $h_{\ell, k}$ are the specific filter coefficients for Layer $\ell$, and $z^{(\ell)}$ is the linear output of the filter,,. The final GNN output is $x^{(L)}$.

## Question 19 (2 points). Draw a diagram of a graph neural network with 3 layers and single features.

A GNN with 3 layers and single features composes three sequential layers, where each layer is a perceptron composed of a graph filter followed by a pointwise nonlinearity $\sigma$,.

**Diagram Structure (based on):**

| Component | Layer 1 ($\ell=1$) | Layer 2 ($\ell=2$) | Layer 3 ($\ell=3$) |
| :---: | :---: | :---: | :---: |
| **Input** | $x^{(0)} = x$ | $x^{(1)}$ | $x^{(2)}$ |
| **Filter Output** | $z^{(1)} = \sum_{k=0}^{K-1} h_{1k} S^k x^{(0)}$ | $z^{(2)} = \sum_{k=0}^{K-1} h_{2k} S^k x^{(1)}$ | $z^{(3)} = \sum_{k=0}^{K-1} h_{3k} S^k x^{(2)}$ |
| **Nonlinearity** | $x^{(1)} = \sigma[z^{(1)}]$ | $x^{(2)} = \sigma[z^{(2)}]$ | $x^{(3)} = \sigma[z^{(3)}]$ |
| **Output** | $x^{(1)}$ | $x^{(2)}$ | $x^{(3)} = \Phi(x; S, H)$ |

The flow is a cascade:
$$x^{(0)} \longrightarrow \fbox{Filter 1 ($h_1$)} \longrightarrow z^{(1)} \longrightarrow \fbox{$\sigma$} \longrightarrow x^{(1)} \longrightarrow \fbox{Filter 2 ($h_2$)} \longrightarrow z^{(2)} \longrightarrow \fbox{$\sigma$} \longrightarrow x^{(2)} \longrightarrow \fbox{Filter 3 ($h_3$)} \longrightarrow z^{(3)} \longrightarrow \fbox{$\sigma$} \longrightarrow x^{(3)}$$

## Question 20 (2 points). Explain how a GNN can be transferred across different graphs.

Transferability refers to the ability to execute a Graph Neural Network (GNN) trained on one graph ($S$) onto a new, different graph ($\tilde{S}$), without needing to retrain the filter coefficients,,.

1.  **GNN Parameterization:** The GNN function $\Phi$ is fundamentally parameterized by the **filter tensor $\mathcal{H}$** (the learned filter coefficients across all layers) and the **graph shift operator $S$**,.
2.  **Training/Filter Tensor:** Once a GNN is trained (by minimizing the Empirical Risk), the resulting output is a fixed, optimal filter tensor $\mathcal{H}^*$,.
3.  **Transference Mechanism:** While $S$ is typically viewed as fixed prior information during training, the transference mechanism reinterprets the shift operator $S$ as an **input** to the GNN operator,.
4.  **Execution on New Graph:** The same learned filter tensor $\mathcal{H}^*$ can then be instantiated using a different graph shift operator $\tilde{S}$ to produce the output $\Phi(x; \tilde{S}, \mathcal{H}^*)$,.

The expressions for graph convolution layers remain the same regardless of the graph,. This inherent separation between the learned coefficients $\mathcal{H}$ and the graph structure $S$ allows the filter (or GNN) to be **transferred** across graphs, even if they have different numbers of nodes, different neighborhoods, or different weights,,. This ability is highly important because identical graphs are rarely encountered in practice.

## Question 21 (2 points). GNNs are particular cases of FCNNs. Why? This fact implies that FCNNs do better than GNNs in the training set. Explain. Is this always a good thing?

**GNNs as Particular Cases of FCNNs:**

A GNN (Graph Neural Network) is a particular case of a Fully Connected Neural Network (FCNN) because the linear maps used in the layers of a GNN are **restricted to be graph convolutional filters**.

1.  An **FCNN** utilizes **arbitrary linear transformations $H$** (generic linear maps) in each layer.
2.  A **GNN** utilizes **graph convolutional filters** in each layer. Since a graph convolutional filter is a specific polynomial on the shift operator $S$ ($\sum h_k S^k x$), it is a restrictive linear map.

Consequently, the optimization set (function class) of the GNN is a **subset** of the optimization set of the FCNN.

**Implication for the Training Set:**

Because the FCNN searches over a larger optimization space than the GNN, the minimization of the empirical risk (loss averaged over the training set $\mathcal{T}$) achieved by the FCNN is necessarily less than or equal to the minimum loss achieved by the GNN.

$$\min_H \sum_{(x,y) \in \mathcal{T}} \mathcal{L}(\Phi(x;H), y) \leq \min_H \sum_{(x,y) \in \mathcal{T}} \mathcal{L}(\Phi(x; S, H), y)$$.

This implies that the fully connected neural network **does better in the training set**.

**Is this always a good thing?**

**No**, achieving a lower loss on the training set is not always a good thing.

1.  The reduction in MSE (Mean Squared Error) achieved by the FCNN can be **illusory** because the FCNN typically **fails to generalize** to signals or examples outside the training set (test set).
2.  The GNN, despite potentially having a higher training error, performs better during live operation (generalization) because it successfully exploits the **internal symmetries** of graph signals codified by the graph shift operator.

Experiments show that FCNNs achieve lower training MSE but perform poorly on the test set, whereas GNNs, though having a slightly higher training MSE, generalize well.

## Question 22 (2 points). Define a graph filter bank. Explain graph filter banks in the GFT domain. Filter banks scatter energy across multiple outputs. Explain.

**Definition of a Graph Filter Bank:**

A **graph filter bank** is defined as a collection of $F$ graph filters applied to a single input signal $x$. Each filter $f$ uses its own set of coefficients $\{h^f_k\}_{k=0}^{K-1}$ to produce an output signal $z^f$. The filter bank produces a collection of $F$ graph signals, which collectively form a **matrix graph signal** $Z = [z^1, z^2, \dots, z^F]$.

**Explanation in the GFT Domain:**

In the Graph Fourier Transform (GFT) domain, graph filters admit a pointwise representation. A filter bank isolates **groups of frequency components**. The energy passed by the output signal $z^f$ of filter $f$ is given by the formula:
$$\|z^f\|^2 = \sum_{i=1}^n \left( h̃^f(\lambda_i) \tilde{x}_i \right)^2$$.
This expression shows that the energy is determined by the summation of weighted squared GFT components of the input signal $\tilde{x}_i$, where the weight is the filter's frequency response $h̃^f(\lambda_i)$ evaluated at the corresponding eigenvalue $\lambda_i$.

**Filter Banks Scatter Energy:**

Filter banks **scatter the energy** of the input signal $x$ into the signals $z^f$ at the output of the filters. They are useful because they identify frequency signatures.

1.  Each filter $f$ in the bank is designed to **pick up energy** concentrated in different GFT components (spectral signatures).
2.  The energy of the input signal is distributed across the multiple output signals $z^f$ based on which filter's response $h̃^f(\lambda)$ overlaps with the signal's spectral signature $\tilde{x}$.
3.  By comparing the energy accumulated at the output of different filters, one can identify signals with different spectral signatures.

## Question 23 (2 points). Define a MIMO graph filter as a collection of filters. Define a MIMO graph filter using matrix notation.

**Definition as a Collection of Filters (MIMO):**

A Multiple-Input-Multiple-Output (MIMO) graph filter processes an input matrix signal $X$ with $F$ input features/signals ($x^1, \dots, x^F$) to produce an output matrix signal $Z$ with $G$ output features/signals ($z^1, \dots, z^G$).

1.  **Intermediate Outputs:** Each input feature $x^f$ is processed by $G$ filters, $h_{fg, k}$, generating intermediate graph signals $u^{fg}$.
2.  **Output Signals:** The output feature $z^g$ is formed by summing the intermediate outputs $u^{fg}$ across all input features $f$:
    $$z^g = \sum_{f=1}^F \left( \sum_{k=0}^{K-1} h^f_{g, k} S^k x^f \right)$$.
The MIMO filter is structurally a collection of $F \times G$ filters.

**Definition using Matrix Notation:**

Using matrix coefficients $H_k \in \mathbb{R}^{F \times G}$, where the entry $(H_k)_{fg}$ is the filter coefficient $h_{fg, k}$, the MIMO graph filter is represented concisely as a polynomial series on the shift operator $S$ that multiplies the input matrix $X$ and the coefficient matrix $H_k$:
$$Z = \sum_{k=0}^{K-1} S^k X H_k$$.
This compact expression is equivalent to the collection-of-filters definition.

## Question 24 (2 points). Give a recursive definition for a GNN with multiple features. Draw and describe a block diagram for a multiple feature GNN with 3 layers. The GNN has F0, F1, F2, and F3 features at layers 0 (the input) 1, 2, and 3 (the output).

**Recursive Definition for a MIMO GNN:**

A GNN with multiple features (MIMO GNN) is constructed by recursively composing MIMO perceptrons. A generic Layer $\ell$ processes the output $X^{\ell-1}$ of the previous layer using a MIMO filter (defined by matrix coefficients $H^{\ell}_k$) followed by a pointwise nonlinearity $\sigma$.

The recursive definition is:
$$X^\ell = \sigma \left[ Z^\ell \right] = \sigma \left[ \sum_{k=0}^{K-1} S^k X^{\ell-1} H_k^\ell \right]$$.

Where:
*   $X^{\ell}$ is the output of layer $\ell$ (a matrix graph signal with $F_\ell$ features).
*   $X^{\ell-1}$ is the input to layer $\ell$ (with $F_{\ell-1}$ features).
*   $X^0 = X$ is the input matrix signal (Layer 0, with $F_0$ features).
*   The final output is $X^L = \Phi(X; S, \mathcal{H})$.

**Block Diagram and Description (3 Layers, $F_0 \to F_1 \to F_2 \to F_3$):**

The MIMO GNN is a cascade where the output of each layer is fed as input to the next layer, with the number of features changing between layers via the MIMO filter coefficients.

| Layer | Input Features | Output Features | Filter Matrix Size | Layer Recursion |
| :---: | :---: | :---: | :---: | :---: |
| 1 | $F_0$ | $F_1$ | $H^1_k: F_0 \times F_1$ | $X^1 = \sigma[Z^1]$, where $Z^1 = \sum_{k=0}^{K-1} S^k X^0 H^1_k$. |
| 2 | $F_1$ | $F_2$ | $H^2_k: F_1 \times F_2$ | $X^2 = \sigma[Z^2]$, where $Z^2 = \sum_{k=0}^{K-1} S^k X^1 H^2_k$. |
| 3 | $F_2$ | $F_3$ | $H^3_k: F_2 \times F_3$ | $X^3 = \sigma[Z^3]$, where $Z^3 = \sum_{k=0}^{K-1} S^k X^2 H^3_k$. |

**Diagram Structure (Conceptual Block Flow):**

$$X^0 (F_0) \xrightarrow{Z^1 = \sum S^k X^0 H^1_k} Z^1 \xrightarrow{\sigma} X^1 (F_1) \xrightarrow{Z^2 = \sum S^k X^1 H^2_k} Z^2 \xrightarrow{\sigma} X^2 (F_2) \xrightarrow{Z^3 = \sum S^k X^2 H^3_k} Z^3 \xrightarrow{\sigma} X^3 (F_3)$$.

## Question 25 (2 points). Define permutation equivariance of a linear operator. Prove that graph filters are permutation equivariant.

**Definition of Permutation Equivariance (Linear Operator):**

Permutation equivariance describes how an operator behaves when the labels (indices) of the nodes are consistently changed. A linear operator $H(S)$ is permutation equivariant if permuting the input signal $x$ and consistently permuting the shift operator $S$ using a permutation matrix $P$ results in the output being permuted by the same operation $P^T$.

Let $\hat{S} = P^T S P$ be the permuted shift operator and $\hat{x} = P^T x$ be the permuted input signal. The condition for equivariance is:
$$H(\hat{S})\hat{x} = P^T H(S)x$$.

**Proof that Graph Filters are Permutation Equivariant:**

The graph filter $H(S)$ is a polynomial on the shift operator $S$: $H(S) = \sum_{k=0}^{K-1} h_k S^k$.

1.  Start with the filter applied to the permuted signal $\hat{x}$ and shift $\hat{S}$:
    $$H(\hat{S})\hat{x} = \sum_{k=0}^{K-1} h_k \hat{S}^k \hat{x}$$.
2.  Substitute the permutation definitions: $\hat{S} = P^T S P$ and $\hat{x} = P^T x$:
    $$H(\hat{S})\hat{x} = \sum_{k=0}^{K-1} h_k (P^T S P)^k P^T x$$.
3.  Expand the matrix power. Since $P P^T = I$ (P and $P^T$ cancel internally):
    $$(P^T S P)^k = P^T S^k P$$.
4.  Substitute the expanded power back into the expression, noting that $P P^T$ cancels:
    $$H(\hat{S})\hat{x} = \sum_{k=0}^{K-1} h_k P^T S^k P P^T x = \sum_{k=0}^{K-1} h_k P^T S^k x$$.
5.  Factor $P^T$ out of the summation:
    $$H(\hat{S})\hat{x} = P^T \left[ \sum_{k=0}^{K-1} h_k S^k x \right]$$.
6.  Identify the original filter operation $H(S)x$ inside the brackets:
    $$H(\hat{S})\hat{x} = P^T H(S)x$$.

This proves that graph filters are **equivariant to permutations**.

## Question 26 (2 points). Define permutation equivariance of an operator. Prove that graph filters are permutation equivariant and that the same is true of GNNs.

**Definition of Permutation Equivariance (Operator):**

An operator $\Phi$ (which may be linear or nonlinear) is permutation equivariant if a consistent permutation of the input signal $x$ and the graph shift operator $S$ results in the output being consistently permuted.

Let $P$ be a permutation matrix, $\hat{x} = P^T x$, and $\hat{S} = P^T S P$. The operator $\Phi$ (which depends on the shift $S$ and filter tensor $\mathcal{H}$) is equivariant if:
$$\Phi(\hat{x}; \hat{S}, \mathcal{H}) = P^T \Phi(x; S, \mathcal{H})$$.

**Proof that Graph Filters are Permutation Equivariant:**

(See Question 25, steps 1–6. Graph filters $H(S)$ are linear operators that satisfy the general equivariance property: $H(\hat{S})\hat{x} = P^T H(S)x$).

**Proof that GNNs are Permutation Equivariant:**

The GNN $\Phi$ is a cascade of $L$ layers, each being a perceptron combining a graph filter $H^\ell(S)$ and a pointwise nonlinearity $\sigma$. The proof proceeds by induction across layers:

1.  **GNN Layer Recursion:** Layer $\ell$ output is $x^\ell = \sigma[H^\ell(S) x^{\ell-1}]$.
2.  **Base Case ($\ell=1$):** The input $x^0=x$ and $\hat{x}^0=\hat{x}$ satisfy $\hat{x}^0 = P^T x^0$ by definition.
3.  **Induction Step:** Assume the inputs to layer $\ell$ satisfy $\hat{x}^{\ell-1} = P^T x^{\ell-1}$.
    *   The permuted layer output is $\hat{x}^\ell = \sigma[H^\ell(\hat{S})\hat{x}^{\ell-1}]$.
    *   Since $H^\ell(S)$ is permutation equivariant (proved above), the filter output term satisfies $H^\ell(\hat{S})\hat{x}^{\ell-1} = P^T H^\ell(S)x^{\ell-1}$.
    *   Substitute this into the expression for $\hat{x}^\ell$:
        $$\hat{x}^\ell = \sigma[P^T H^\ell(S)x^{\ell-1}]$$.
    *   Since $\sigma$ is a **pointwise nonlinearity**, it commutes with the permutation matrix $P^T$: $\sigma[P^T z] = P^T \sigma[z]$.
        $$\hat{x}^\ell = P^T \sigma[H^\ell(S)x^{\ell-1}] = P^T x^\ell$$.
4.  **Conclusion:** The layer output $\hat{x}^\ell$ is consistently permuted ($P^T x^\ell$), completing the induction step. Since this holds for all layers, the entire GNN operator $\Phi$ is permutation equivariant.

## Question 27 (2 points). GNNs and graph filters perform label-independent processing of graph signals. Explain. Drawing an example may help with your explanation, but you are required to write down equations to justify your claims.

**Explanation:**

GNNs and graph filters perform **label-independent processing** because their computations depend exclusively on the underlying connectivity and weights defined by the shift operator $S$ and the signal values $x$, and not on the arbitrary numerical labels (indices) assigned to the nodes. If the physical graph and signal characteristics are maintained, changing the node labels does not alter the output's intrinsic structure.

**Justification (Equations):**

A change in node labels is mathematically represented by applying a permutation matrix $P$ to the signal $x$ and consistently transforming the shift operator $S$ into $\hat{S}$:
*   $\hat{x} = P^T x$ (Relabeled input signal)
*   $\hat{S} = P^T S P$ (Relabeled shift operator)

The output of a GNN or graph filter, $\Phi(x; S, H)$, must satisfy the **Permutation Equivariance Theorem**:

$$\Phi(\hat{x}; \hat{S}, H) = P^T \Phi(x; S, H)$$.

The left side, $\Phi(\hat{x}; \hat{S}, H)$, represents the result of processing the relabeled graph and signal. The right side, $P^T \Phi(x; S, H)$, shows that this result is simply the **relabeled version ($P^T$) of the original output**. This proves that the core processing is independent of the labeling used.

## Question 28 (2 points). GNNs and graph filters leverage internal symmetries. Explain. Drawing an example may help with your explanation, but you are required to write down equations to justify your claims.

**Explanation:**

GNNs and graph filters leverage **internal symmetries** by exploiting situations where the graph itself is invariant under a specific permutation $P$. This means that once the system learns how to process an input signal $x$, it implicitly knows how to process any permuted version $P^T x$ that maintains the graph's symmetric structure, effectively **multiplying the perceived size of the dataset**.

**Justification (Equations):**

A graph possesses symmetry under permutation $P$ if the shift operator $S$ maps onto itself:
$$S = P^T S P$$.

When this symmetry condition holds, the Permutation Equivariance theorem implies the following relationship for the GNN operator $\Phi$:

$$\Phi(P^T x; S, H) = \Phi(P^T x; P^T S P, H) = P^T \Phi(x; S, H)$$.

The resulting equation, $\Phi(P^T x; S, H) = P^T \Phi(x; S, H)$, shows that the output obtained by processing the permuted input $P^T x$ is simply the permuted output $P^T \Phi(x; S, H)$. This means the GNN learns how to process the signal $P^T x$ by observing only $x$, successfully exploiting the graph's internal symmetry.

## Question 29 (2 points). Define Lipschitz filters. Define integral Lipschitz filters. Compare their different discriminability properties.

**Definition of Lipschitz Filters:**

A graph filter is **Lipschitz** if its graph frequency response $h̃(\lambda)$ satisfies:
$$\left| h̃(\lambda_2) - h̃(\lambda_1) \right| \leq C \left| \lambda_2 - \lambda_1 \right|$$
for some finite constant $C > 0$ and any $\lambda_1, \lambda_2$. This condition bounds the derivative (slope) of the frequency response: $|h̃'(\lambda)| \leq C$.

**Definition of Integral Lipschitz Filters:**

A graph filter is **integral Lipschitz** if its graph frequency response $h̃(\lambda)$ satisfies:
$$\left| h̃(\lambda_2) - h̃(\lambda_1) \right| \leq C \left| \lambda_2 - \lambda_1 \right| \frac{\left| \lambda_1 + \lambda_2 \right|}{2}$$
for some finite constant $C > 0$ and any $\lambda_1, \lambda_2$. This condition implies a bound on the product of the frequency and the derivative: $|\lambda h̃'(\lambda)| \leq C$.

**Comparison of Discriminability Properties:**

Discriminability is related to how sharply the frequency response $h̃(\lambda)$ can vary, which is governed by the constant $C$.

| Property | Lipschitz Filters | Integral Lipschitz Filters |
| :--- | :--- | :--- |
| **Low Frequencies ($\lambda \approx 0$)| High discriminability possible by maximizing $C$, leading to filters that vary rapidly. | Can achieve **arbitrarily high discriminability** (very thin filters) around $\lambda=0$, regardless of the constant $C$. |
| **High Frequencies (Large $\lambda$)| Can be highly discriminative and sharp (high slope) if $C$ is large enough. | Must be **flat** and lose discriminability, regardless of how large $C$ is chosen. They cannot discriminate high frequency features. |

## Question 30 (2 points). State a theorem claiming the stability of integral Lipschitz graph filters to scaling of a graph. This theorem implies that graph filters cannot be discriminative and stable at the same time. Explain.

**Theorem Statement (Stability to Scaling):**

The theorem states that **Integral Lipschitz Graph Filters are Stable to Scaling**.

Given graph shift operators $S$ and $\hat{S} = (1 + \varepsilon) S$ (where $\varepsilon$ scales the edges), and an integral Lipschitz filter $H$ with constant $C$:
$$\|H(\hat{S}) - H(S)\| \leq C \varepsilon + O(\varepsilon^2)$$.

This means the difference between the filters is bounded proportionally to the scaling factor $\varepsilon$ and the integral Lipschitz constant $C$.

**Implication: Graph Filters Cannot Be Discriminative and Stable Simultaneously:**

1.  **Requirement for Stability:** The theorem shows that stability to scaling requires the filter to be **integral Lipschitz**.
2.  **Constraint on Discriminability:** Integral Lipschitz filters are characterized by the constraint that their frequency response $h̃(\lambda)$ must be **flat at high frequencies**. This is because the integral Lipschitz condition, $|\lambda h̃'(\lambda)| \leq C$, forces the derivative to vanish at least linearly as $\lambda$ increases.
3.  **Incompatibility:** A flat response at high frequencies means the filter cannot discriminate high-frequency features (high discriminability implies high slope). Therefore, since stability demands the constraints of integral Lipschitz filters, and those constraints preclude high-frequency discrimination, stability and high-frequency discriminability are **plainly incompatible** for linear graph filters.

## Question 31 (2 points). State a theorem claiming the stability of GNNs with layers made up of integral Lipschitz filters to scaling of a graph. This theorem implies that GNNs can be discriminative and stable at the same time. Explain

#### Stability Theorem Statement
The stability of GNNs with layers composed of integral Lipschitz filters to scaling of the graph shift operator is stated by the following theorem:

**Theorem (Integral Lipschitz GNNs are Stable to Scaling):** Given shift operators $S$ and $\hat{S} = (1 + \varepsilon) S$ and a GNN operator $\Phi(\cdot; S, H)$ with $L$ single-feature layers. If the filters at each layer have **unit operator norms** and are **integral Lipschitz with constant $C$**, and the nonlinearity $\sigma$ is **normalized Lipschitz**, then the operator distance modulo permutation is bounded by:
$$\|\Phi(\cdot; \hat{S}, H) - \Phi(\cdot; S, H)\| \leq C L \varepsilon + O(\varepsilon^2)$$.

This theorem demonstrates that the stability property is **inherited** from the integral Lipschitz filters and **accumulates linearly** across the $L$ layers.

#### Implication: GNNs can be Discriminative and Stable Simultaneously
For linear graph filters, stability (requiring integral Lipschitz filters) is fundamentally **incompatible** with high-frequency discriminability, because integral Lipschitz filters must have a **flat (non-discriminative) frequency response** at high frequencies ($|\lambda h̃'(\lambda)| \leq C$).

GNNs overcome this limitation due to the **pointwise nonlinearity ($\sigma$)** used in each layer.

1.  **Stability is Maintained:** The stability property is inherited from the integral Lipschitz filters.
2.  **Discriminability is Enhanced:** The nonlinearity $\sigma$ acts as a **low-pass demodulator** which reduces variability and moves high-frequency components of the signal **into the low-frequency domain**.
3.  **Exploiting Low Frequencies:** Since integral Lipschitz filters can be **very sharp** (highly discriminative) and **very stable** at low frequencies ($\lambda \approx 0$) regardless of the constant $C$, the low-frequency components demodulated by $\sigma$ can be successfully discriminated by the filter in the *next layer*.

Therefore, GNNs achieve a **legitimate tradeoff** where they are stable (due to integral Lipschitz filters) and discriminative (due to low-pass demodulation by $\sigma$), a capability that linear graph filters cannot attain.

## Question 32 (2 points). Prove that integral Lipschitz graph filters are stable to the scaling of a graph and that GNNs with layers made up of integral Lipschitz filters inherit this property.

#### Part 1: Proof that Integral Lipschitz Graph Filters are Stable to Scaling

This proof relies on showing that the filter difference norm is bounded by $C\varepsilon$. We assume the shift operators are related by $\hat{S} = (1 + \varepsilon) S$.

1.  **Filter Variation $\Delta(S)$:** The difference between $H(\hat{S})$ and $H(S)$ is approximated to first order by the filter variation $\Delta(S)$, where $H(\hat{S}) - H(S) = \Delta(S) + O(\varepsilon^2)$. The filter variation is defined as:
    $$\Delta(S) = \varepsilon \sum_{k=0}^{\infty} k h_k S^k$$.
2.  **GFT Domain Shift:** We evaluate $\|\Delta(S)x\|$ for a unit norm vector $x$. By projecting $x$ onto the eigenvector basis ($v_i$) of $S$ and using the eigenvector property $S^k v_i = \lambda_i^k v_i$, the expression simplifies to:
    $$\Delta(S)x = \varepsilon \sum_{i=1}^n \tilde{x}_i \left( \lambda_i h'(\lambda_i) \right) v_i$$
    where $ \lambda_i h'(\lambda_i) = \sum_{k=0}^{\infty} k h_k \lambda_i^k$.
3.  **Integral Lipschitz Bound:** The stability is proven by computing the squared norm and applying the integral Lipschitz condition, which bounds the term in parentheses by $C$: $|\lambda_i h'(\lambda_i)| \leq C$.
    $$\|\Delta(S)x\|^2 = \varepsilon^2 \sum_{i=1}^n \tilde{x}_i^2 \left( \lambda_i h'(\lambda_i) \right)^2 \leq \varepsilon^2 \sum_{i=1}^n \tilde{x}_i^2 C^2$$.
4.  **Conclusion:** Since the GFT is unitary, $\sum \tilde{x}_i^2 = \|x\|^2 = 1$, yielding $\|\Delta(S)x\| \leq C\varepsilon$. Thus, $\|\mathbf{H}(\hat{S}) - \mathbf{H}(S)\| \leq C \varepsilon + O(\varepsilon^2)$.

#### Part 2: Proof that GNNs Inherit this Property

The proof of GNN stability proceeds by induction over the $L$ layers, leveraging the **pointwise nature** of the nonlinearity $\sigma$.

1.  **Eliminating Nonlinearity:** Let $x^\ell$ and $\hat{x}^\ell$ be the outputs of Layer $\ell$ running on $S$ and $\hat{S}$, respectively. Since the nonlinearity $\sigma$ is **normalized Lipschitz** ($\|\sigma(x_2) - \sigma(x_1)\| \leq \|x_2 - x_1\|$) and acts componentwise, the difference in outputs is bounded by the difference in the filter outputs $z^\ell$ and $\hat{z}^\ell$:
    $$\|x^\ell - \hat{x}^\ell\| \leq \|H^\ell(S)x^{\ell-1} - H^\ell(\hat{S})\hat{x}^{\ell-1}\|$$.
2.  **Norm Manipulation (Layer Bound):** By adding and subtracting $H^\ell(\hat{S})x^{\ell-1}$ and using the triangle inequality:
    $$\|x^\ell - \hat{x}^\ell\| \leq \|H^\ell(S) - H^\ell(\hat{S})\| \|x^{\ell-1}\| + \|H^\ell(\hat{S})\| \|x^{\ell-1} - \hat{x}^{\ell-1}\| + O(\varepsilon^2)$$.
3.  **Recursive Application:** Using the normalized assumptions ($\|H^\ell(\hat{S})\|=1$, $\|x^{\ell-1}\|\leq 1$) and the filter stability proved in Part 1 ($\|H^\ell(S) - H^\ell(\hat{S})\| \leq C\varepsilon$):
    $$\|x^\ell - \hat{x}^\ell\| \leq C\varepsilon + \|x^{\ell-1} - \hat{x}^{\ell-1}\| + O(\varepsilon^2)$$.
4.  **Accumulation:** Recursively applying this bound from Layer $L$ back to Layer 1 (where the input difference $\|x^0 - \hat{x}^0\|$ is zero) results in the accumulation of the $C\varepsilon$ distortion term across $L$ layers, yielding the final bound $C L \varepsilon$.

## Question 33 (2 points). Prove that integral Lipschitz graph filters are stable to the scaling of a graph.

The stability theorem for integral Lipschitz graph filters states that for shift operators $S$ and $\hat{S} = (1 + \varepsilon) S$, the filter difference is bounded by $\|\mathbf{H}(\hat{S})-\mathbf{H}(S)\| \leq C \varepsilon + O(\varepsilon^2)$.

The core of the proof lies in relating the filter variation to the derivative of the frequency response and utilizing the integral Lipschitz condition.

1.  **Filter Variation $\Delta(S)$:** Express the filter difference $H(\hat{S}) - H(S)$ using the polynomial definition and the binomial approximation of $(1 + \varepsilon)^k \approx 1 + k\varepsilon$:
    $$H(\hat{S}) - H(S) = \varepsilon \sum_{k=0}^{\infty} k h_k S^k + O(\varepsilon^2)$$.
    The relevant variation term is $\Delta(S) = \varepsilon \sum_{k=0}^{\infty} k h_k S^k$.
2.  **Shifting to GFT Domain and Derivative Recognition:** Applying $\Delta(S)$ to a unit norm signal $x$ and shifting to the GFT domain shows that the operation is equivalent to multiplying the GFT components $\tilde{x}_i$ by the term $\lambda_i h'(\lambda_i)$:
    $$\Delta(S)x = \varepsilon \sum_{i=1}^n \tilde{x}_i \left( \lambda_i h'(\lambda_i) \right) v_i$$.
3.  **Bounding the Norm using Integral Lipschitz:** Since the integral Lipschitz condition dictates $|\lambda_i h'(\lambda_i)| \leq C$, and $x$ has unit norm ($\sum \tilde{x}_i^2 = 1$), computing the squared norm and applying the bound proves the required stability:
    $$\|\Delta(S)x\|^2 = \varepsilon^2 \sum_{i=1}^n \tilde{x}_i^2 \left( \lambda_i h'(\lambda_i) \right)^2 \leq \varepsilon^2 \sum_{i=1}^n \tilde{x}_i^2 C^2 = (C\varepsilon)^2$$.

## Question 34 (2 points). Define additive perturbations of graph shift operators. Introduce distances modulo permutation written explicitly in terms of an error matrix E. Define the eigenvector misalignment constant $\delta$ of an additive perturbation model.

#### Additive Perturbations of Graph Shift Operators
An **additive perturbation model** relates the perturbed graph shift operator $\hat{S}$ to the original shift operator $S$ by the addition of an error matrix $E$:
$$\hat{S} = S + E$$.

#### Distances Modulo Permutation
Since graphs related by a permutation $P$ are considered the same (relabeling), the distance between $S$ and $\hat{S}$ must account for all possible relabelings.

1.  The set of all possible error matrices $\tilde{E}$ relating $S$ to a permuted version of $\hat{S}$ is defined as:
    $$\mathcal{E}(S, \hat{S}) = \{ \tilde{E} : P^T \hat{S} P = S + \tilde{E}, P \in \mathcal{P} \}$$.
2.  The **error matrix modulo permutation** $E$ is the element in $\mathcal{E}(S, \hat{S})$ with the minimum operator norm:
    $$E = \text{argmin}_{\tilde{E} \in \mathcal{E}(S, \hat{S})} \|\tilde{E}\|$$.
3.  The **operator distance modulo permutation** $d(S, \hat{S})$ is then defined explicitly in terms of the norm of this minimal error matrix $E$:
    $$d(S, \hat{S}) = \|E\| = \min_{\tilde{E} \in \mathcal{E}(S, \hat{S})} \|\tilde{E}\|$$.

#### Eigenvector Misalignment Constant $\delta$
The **eigenvector misalignment constant $\delta$** quantifies the difference between the eigenvectors of the shift operator $S$ (matrix $V$) and the error matrix $E$ (matrix $U$):
$$\delta = \left[\left(\|U - V\| + 1\right)^2 - 1\right]$$.
Because $U$ and $V$ are unitary matrices with norms $\leq 1$, this constant $\delta$ is bounded, satisfying $\delta \leq 8$.

## Question 35 (2 points). Define multiplicative (relative) perturbations of graph shift operators. Introduce distances modulo permutation written explicitly in terms of an error matrix E. Define the eigenvector misalignment constant $\delta$ of a multiplicative perturbation model.

#### Multiplicative (Relative) Perturbations of Graph Shift Operators
A **multiplicative or relative perturbation model** relates the perturbed shift operator $\hat{S}$ to $S$ using symmetric error terms involving the error matrix $E$.
$$\hat{S} = S + E S + S E$$.
This model is considered more meaningful because it ties changes in edge weights to the local structure of the graph.

#### Distances Modulo Permutation
For the relative model, the perturbation must also be considered modulo permutation $P$:

1.  The set of relative error matrices modulo permutation is defined using the condition $P^T \hat{S} P = S + \tilde{E} S + S \tilde{E}$.
2.  The **relative error matrix modulo permutation** $E$ is the matrix $\tilde{E}$ in this set with the smallest operator norm:
    $$E = \text{argmin}_{\tilde{E} \in \mathcal{E}(S, \hat{S})} \|\tilde{E}\|$$.
3.  The **relative distance modulo permutation** $d(S, \hat{S})$ is defined by the norm of this minimal error matrix $E$:
    $$d(S, \hat{S}) = \|E\|$$.

#### Eigenvector Misalignment Constant $\delta$
The **eigenvector misalignment constant $\delta$** is defined identically to the additive case, comparing the eigenvector matrices $V$ (of $S$) and $U$ (of $E$):
$$\delta = \left[\left(\|U - V\| + 1\right)^2 - 1\right]$$.

## Question 36 (2 points). State a theorem claiming the stability of Lipschitz filters to additive perturbations. Explain the relevant constants that appear in the bound. The bound claims stability, which is stronger than continuity. Explain. The bound is universal for all graphs with a given number of nodes. Explain.

#### Theorem Statement (Lipschitz Filters are Stable to Additive Perturbations)
**Theorem:** Consider a graph filter $h$ along with shift operators $S$ and $\hat{S}$ having $n$ nodes. If the shift operators are related by $P^T \hat{S} P = S + E$, the error matrix $E$ has norm $\|E\| = \varepsilon$ and eigenvector misalignment $\delta$ relative to $S$, and the filter $h$ is **Lipschitz with constant $C$**. Then, the operator distance modulo permutation between filters $H(S)$ and $H(\hat{S})$ is bounded by:
$$\|\mathbf{H}(\hat{S}) - \mathbf{H}(S)\|_{\mathcal{P}} \leq C \left(1 + \frac{\delta}{\sqrt{n}}\right) \varepsilon + O(\varepsilon^2)$$.

#### Explanation of Relevant Constants

The bound $C \left(1 + \frac{\delta}{\sqrt{n}}\right) \varepsilon$ contains three key components:

1.  **$C$ (The Lipschitz Constant):** This constant represents the maximum slope of the filter's frequency response, and is determined by the filter coefficients chosen by the designer. $C$ controls the **discriminability** of the filter: a larger $C$ increases discriminability but makes the filter more sensitive (less stable) to perturbations.
2.  **$\varepsilon$ (The Perturbation Size):** This is the norm $\|E\|$ of the error matrix, quantifying the distance between $S$ and $\hat{S}$ modulo permutation.
3.  **$\left(1 + \frac{\delta}{\sqrt{n}}\right)$ (The Misalignment Factor):** This factor depends on the **eigenvector misalignment constant $\delta$** (a property of the perturbation $E$) and the number of nodes $n$. This factor is **uncontrollable** by the system designer through filter choice.

#### Stability vs. Continuity
The bound claims **stability**, which is stronger than plain continuity.

*   **Continuity** implies that small input changes ($\varepsilon$) lead to small output changes.
*   **Stability** (Lipschitz continuity) implies that the output changes are directly **proportional** to the input perturbation size $\varepsilon$. The relationship is linearly bounded by the overall Lipschitz constant $C \left(1 + \frac{\delta}{\sqrt{n}}\right)$.

#### Universality of the Bound
The bound is **universal** because it holds uniformly for **all graphs** with a given number of nodes $n$. This is because the bound depends only on:
1.  The filter properties (Lipschitz constant $C$).
2.  The perturbation properties ($\varepsilon$ and $\delta$).
3.  The graph size $n$.

Crucially, **no constant in the bound depends on the specific choice of the graph shift operator $S$** itself, beyond its size.

## Question 37 (2 points). State a theorem claiming the stability of Lipschitz filters to additive perturbations. Explain the relevant constants that appear in the bound. The bound is universal for all graphs with a given number of nodes. Explain.

#### Stability Theorem Statement
The theorem states that **Lipschitz filters are stable to additive perturbations** of the graph support.

Given shift operators $S$ and $\hat{S}$ having $n$ nodes, if the filter $h$ is **Lipschitz with constant $C$** and the shift operators are related by $P^T \hat{S} P = S + E$ (where $E$ has norm $\|E\| = \varepsilon$ and eigenvector misalignment $\delta$), the operator distance modulo permutation between the filters is bounded by:
$$\left\|\mathbf{H}(\hat{S})-\mathbf{H}(S)\right\|_{\mathcal{P}} \leq C \left(1+\frac{\delta}{\sqrt{n}}\right) \varepsilon + O(\varepsilon^2)$$.

#### Explanation of Relevant Constants

1.  **$C$ (The Lipschitz Constant):** This constant is determined by the filter coefficients and controls the **discriminability** of the filter. A larger $C$ allows for sharper filters (higher discriminability), but this comes at the cost of stability, making the filter more susceptible to graph perturbations. $C$ is a **controllable design parameter**.
2.  **$\varepsilon$ (The Perturbation Size):** This is the norm of the error matrix, $\|E\|$, which quantifies the distance between $S$ and $\hat{S}$ modulo permutation.
3.  **$\left(1+\frac{\delta}{\sqrt{n}}\right)$ (The Misalignment Factor):** This factor depends on the **eigenvector misalignment constant $\delta$** and the number of nodes $n$. The constant $\delta$ quantifies the difference between the eigenvectors of $S$ and $E$. This factor is **uncontrollable** by the designer, as it depends on the nature of the perturbation itself.

#### Universality of the Bound

The bound is **universal** because it holds uniformly for **all graphs** with a given number of nodes $n$. This is because the bound only relies on:

1.  Properties of the **filter's frequency response** (specifically, $C$).
2.  Properties of the **perturbation matrix** $E$ ($\varepsilon$ and $\delta$).
3.  The **number of nodes** $n$.

Crucially, **no constant in the bound depends on the specific structure of the graph shift operator $S$**, only its size $n$.

## Question 38 (2 points). State a theorem claiming the stability of integral Lipschitz filters to multiplicative (relative) perturbations. Explain the relevant constants that appear in the bound. The bound claims stability, which is stronger than continuity. Explain.

#### Stability Theorem Statement
The theorem states that **Integral Lipschitz filters are stable to relative perturbations** of the graph support.

Given shift operators $S$ and $\hat{S}$ having $n$ nodes, if the filter $h$ is **integral Lipschitz with constant $C$** and the shifts are related by $P^T \hat{S} P = S + ES + SE$ (where $E$ has norm $\|E\| = \varepsilon$ and eigenvector misalignment $\delta$), the operator distance modulo permutation between the filters is bounded by:
$$\left\|\mathbf{H}(\hat{S})-\mathbf{H}(S)\right\|_{\mathcal{P}} \leq 2C \left(1+\frac{\delta}{\sqrt{n}}\right) \varepsilon + O(\varepsilon^2)$$.

#### Explanation of Relevant Constants

1.  **$C$ (The Integral Lipschitz Constant):** This constant governs the behavior of the filter's frequency response, especially near $\lambda=0$ and large $\lambda$. It is a controllable value determined by filter choice.
2.  **$\varepsilon$ (The Relative Perturbation Size):** This is the norm $\|E\|$, which represents the **relative distance modulo permutation** $d(S, \hat{S})$.
3.  **$\left(1+\frac{\delta}{\sqrt{n}}\right)$ (The Misalignment Factor):** This factor depends on the **eigenvector misalignment constant $\delta$** and the number of nodes $n$.

#### Stability is Stronger than Continuity

The bound claims **stability**, which is stronger than plain continuity.

1.  **Continuity** merely implies that if the input change ($\varepsilon$) is small, the output change ($\|\mathbf{H}(\hat{S})-\mathbf{H}(S)\|_{\mathcal{P}}$) is also small.
2.  **Stability** (specifically, Lipschitz continuity) implies a **linear relationship** between input perturbation and output distortion. The distortion in the filter output is **proportional** to the size of the perturbation $\varepsilon$, bounded by the constant $2C \left(1+\frac{\delta}{\sqrt{n}}\right)$.

## Question 39 (2 points). State a theorem claiming the stability of integral Lipschitz filters to multiplicative (relative) perturbations. Explain the relevant constants that appear in the bound. The bound is universal for all graphs with a given number of nodes. Explain.

#### Stability Theorem Statement
The theorem states that **Integral Lipschitz filters are stable to relative perturbations**.

Given shift operators $S$ and $\hat{S}$ having $n$ nodes, if the filter $h$ is **integral Lipschitz with constant $C$** and $P^T \hat{S} P = S + ES + SE$, the operator distance modulo permutation is bounded by:
$$\left\|\mathbf{H}(\hat{S})-\mathbf{H}(S)\right\|_{\mathcal{P}} \leq 2C \left(1+\frac{\delta}{\sqrt{n}}\right) \varepsilon + O(\varepsilon^2)$$.

#### Explanation of Relevant Constants

1.  **$C$ (The Integral Lipschitz Constant):** This constant controls the filter's frequency response profile and is a controllable design parameter.
2.  **$\varepsilon$ (The Relative Perturbation Size):** The norm of the relative error matrix $E$, which measures how far $\hat{S}$ is from being a permutation of $S$ in relative terms.
3.  **$\delta$ (Eigenvector Misalignment Constant):** Measures the similarity between the eigenvectors of the shift operator $S$ and the error matrix $E$.
4.  **$n$ (Number of Nodes):** The size of the graph.

#### Universality of the Bound

The bound is **universal** for all graphs with a given number of nodes $n$.

This universality stems from the fact that the bound's proportionality constants do **not depend on the specific shift operator $S$** itself, but only on properties extrinsic to $S$:
1.  The filter constant $C$ (a property of the frequency response, not the graph).
2.  The perturbation constants $\varepsilon$ and $\delta$ (properties of the error matrix $E$).
3.  The graph size $n$.

## Question 40 (2 points). Additive perturbation models of graphs are not as meaningful as relative (multiplicative) perturbation models of graphs. Explain.

**Additive perturbation models** define the perturbed shift operator as $\hat{S} = S + E$. **Relative perturbation models** define it symmetrically as $P^T \hat{S} P = S + ES + SE$.

Additive models are considered **not meaningful** because they fail to relate the magnitude of the error to the local structure of the graph.

1.  **Issue with Additive Models:** In additive perturbations, the size of the error, measured by $\|E\|$, does not guarantee a meaningful interpretation of the perturbation. For instance, if a graph has edges with both large weights ($W$) and small weights ($w$), a small error norm $\|E\|$ could still lead to **drastic relative changes** in the small $w$ edges, changing the fundamental character of that community, simply because the absolute scale of the perturbation is set by $W$.
2.  **Advantage of Relative Models:** Relative perturbations are **more meaningful** because they tie changes in edge weights to the local connectivity. The perturbation terms ($ES+SE$) ensure that the magnitude of the change in an edge weight is proportional to the degrees of its incident nodes. Consequently, parts of the graph with **weaker connectivity experience smaller changes** than parts with stronger links. This preserves the overall structure and character of the graph in a way that additive perturbations do not.

## Question 41 (2 points). GNNs whose layers are made up of Lipschitz filters are stable to additive deformations of the graph support. State a theorem and explain.

#### Stability Theorem Statement
The theorem states that **GNNs are stable to additive perturbations** if their layers use Lipschitz filters.

Given a GNN $\Phi(\cdot; S, \mathbf{H})$ with $L$ single-feature layers, where all filters are **Lipschitz (constant $C$)** and the nonlinearity $\sigma$ is normalized Lipschitz. If $P^T \hat{S} P = S + E$, the operator distance modulo permutation is bounded by:
$$\left\|\Phi(\cdot; \hat{S}, \mathbf{H})-\Phi(\cdot; S, \mathbf{H})\right\|_{\mathcal{P}} \leq C \left(1+\frac{\delta}{\sqrt{n}}\right) L \varepsilon + O(\varepsilon^2)$$.

#### Explanation
GNNs **inherit the stability property** of the underlying filter class.

1.  **Filter Stability:** The Lipschitz filters provide the base stability, as they are stable to additive perturbations.
2.  **Propagation through Layers ($L$ factor):** The distortion ($\propto \varepsilon$) introduced by the graph deformation in the first layer propagates through subsequent layers. Since the **nonlinearity $\sigma$ is pointwise and normalized Lipschitz**, it is unaware of the graph and **does not amplify the distortion**. Instead, the distortion accumulates linearly, resulting in the factor $L$ appearing in the bound.
3.  **Inheritance:** Because the nonlinearity $\sigma$ is easily handled (it eliminates itself from the distortion analysis due to its normalized Lipschitz property), the GNN's stability behavior is dictated entirely by the stability properties of the linear graph filters in its layers.

## Question 42 (2 points). GNNs whose layers are made up of integral Lipschitz filters are stable to relative (multiplicative) deformations of the graph support. State a theorem and explain.

#### Stability Theorem Statement
The theorem states that **GNNs are stable to relative perturbations** if their layers use integral Lipschitz filters.

Given a GNN $\Phi(\cdot; S, \mathbf{H})$ with $L$ single-feature layers, where all filters are **integral Lipschitz (constant $C$)** and the nonlinearity $\sigma$ is normalized Lipschitz. If $P^T \hat{S} P = S + ES + SE$, the operator distance modulo permutation is bounded by:
$$\left\|\Phi(\cdot; \hat{S}, \mathbf{H})-\Phi(\cdot; S, \mathbf{H})\right\|_{\mathcal{P}} \leq 2C \left(1+\frac{\delta}{\sqrt{n}}\right) L \varepsilon + O(\varepsilon^2)$$.

#### Explanation
This stability is inherited because the filter class required for stability against relative perturbations (integral Lipschitz filters) is successfully preserved across GNN layers.

1.  **Filter Requirement:** Stability to relative perturbations (which are the most meaningful type) **requires** the use of **integral Lipschitz filters**.
2.  **Inheritance:** The GNN inherits this property because the pointwise nonlinearity $\sigma$ **does not contribute to the distortion** caused by the graph deformation. The single-layer distortion (bounded by the integral Lipschitz constant $C$) accumulates linearly across the $L$ layers.
3.  **Role of Nonlinearity:** By being normalized Lipschitz, $\sigma$ bounds the layer-to-layer distortion, ensuring that the GNN's overall stability is dominated by the linear accumulation of distortion caused by the filters' instability to the shift operator change.

## Question 43 (2 points). Explain the stability vs discriminability tradeoff of graph filters. Explain the stability vs discriminability tradeoff of GNNs. Highlight their differences.

#### Graph Filters (GFs) Tradeoff

The tradeoff for GFs depends on the frequency component being processed:

1.  **Stability/Discriminability Incompatibility (Relative Perturbations):** For stability to meaningful (relative) deformations, GFs must be **integral Lipschitz**. Integral Lipschitz filters are constrained to be **flat (non-discriminative) at high frequencies** ($\lambda$). Thus, stability and high-frequency discriminability are **incompatible**.
2.  **Low Frequencies:** GFs can be arbitrarily stable and discriminative at low frequencies.

#### GNNs Tradeoff

GNNs achieve a **legitimate tradeoff** where stability and discriminability are **compatible**.

1.  **Inherited Stability:** The GNN maintains stability by using integral Lipschitz filters in its layers, inheriting the stability property.
2.  **Nonlinearity for Discriminability:** GNNs overcome the filter's high-frequency limitation using the **pointwise nonlinearity ($\sigma$)**, which acts as a **low-pass demodulator**. This process moves high-frequency components of the signal into the low-frequency domain.
3.  **Exploitation:** Once shifted to low frequencies, these components can be sharply discriminated by the stable integral Lipschitz filters in **deeper layers**.

#### Differences Highlighted

The key difference is the role of the nonlinearity:
*   **GFs:** Lacking nonlinearity, GFs are limited by the inherent constraints of their filter class (Integral Lipschitz) and must sacrifice high-frequency discriminability for stability.
*   **GNNs:** The nonlinearity decouples discriminability from the input frequency by transforming high frequencies into stable, low-frequency features that the subsequent filter layers can process accurately.

## Question 44 (2 points). When using a graph filter, higher frequencies are more difficult to process. Explain. Integral Lipschitz filters are stable to deformations because they do not attempt to discriminate high frequency components. Explain.

#### Higher Frequencies are Difficult to Process

Higher frequencies are more difficult to process due to their extreme sensitivity to graph perturbations.

1.  **Eigenvalue Movement:** When the graph shift operator $S$ undergoes a perturbation, such as scaling by $(1 + \varepsilon)$, its **eigenvalues ($\lambda_i$) are also dilated** by $(1 + \varepsilon)$.
2.  **Proportional Distortion:** Since the distortion is proportional to the eigenvalue, high eigenvalues move significantly more than low eigenvalues.
3.  **Filter Sensitivity:** This means that even a small perturbation ($\varepsilon$) yields **large differences** between the intended instantiation of the filter's response $h̃(\lambda_i)$ and the actual instantiated response $h̃(\hat{\lambda}_i)$. This large variability makes processing signals with high-frequency components very difficult, as the filter output becomes highly unstable.

#### Integral Lipschitz Stability

Integral Lipschitz filters achieve stability by limiting the filter's variability at high frequencies.

1.  **Integral Lipschitz Constraint:** The condition $|\lambda h̃'(\lambda)| \leq C$ forces the derivative of the frequency response to **vanish (or be flat) for large $\lambda$**.
2.  **Ignoring Variability:** Stability is attained because the filter **does not attempt to discriminate** high-frequency components. If an eigenvalue moves significantly due to deformation, the filter's response $h̃(\lambda)$ remains nearly constant due to the flatness constraint.
3.  **Result:** By being flat at high frequencies, the filter ensures that the difference between the response evaluated at the ideal eigenvalue and the perturbed eigenvalue is small, thereby achieving stability.

## Question 45 (2 points). We have signals $v_i$ and $v_j$ aligned with eigenvectors $i$ and $j$ of a shift operator. Explain how these signals can be discriminated with graph filters. What happens when the shift operator is perturbed? Your answer must imply that stability and discriminability are incompatible when we use graph filters.

#### Discrimination with Graph Filters
Signals $v_i$ and $v_j$, which align with eigenvectors $i$ and $j$ of the shift operator $S$, correspond to spikes in the Graph Fourier Transform (GFT) domain at their respective eigenvalues $\lambda_i$ and $\lambda_j$.

To discriminate $v_i$ from $v_j$ using a graph filter $H(S)$:
1.  A filter is designed with a frequency response $h̃(\lambda)$ that exhibits high energy (e.g., $h̃(\lambda) \approx 1$) at $\lambda_i$ and low energy (e.g., $h̃(\lambda) \approx 0$) at $\lambda_j$.
2.  When the input signal is $v_i$, the output signal $y_i = H(S)v_i$ has high energy.
3.  When the input signal is $v_j$, the output signal $y_j = H(S)v_j$ has low energy.
4.  By comparing the output energies of two such filters (one designed for $v_i$ and one for $v_j$), the signals can be discriminated.

#### Effect of Perturbation and Incompatibility
If the shift operator $S$ is perturbed (e.g., scaled by $1 + \varepsilon$), the eigenvalues also dilate ($\lambda_i \to \hat{\lambda}_i$). Assuming $\lambda_i$ and $\lambda_j$ are high frequencies, this movement is significant.

1.  **Discriminability Requires Sharpness:** For the filter to be highly discriminative, its frequency response must be sharp (i.e., have a large slope or Lipschitz constant $C$).
2.  **Perturbation Causes Failure:** If the filter is highly discriminative but runs on the perturbed graph $\hat{S}$, the response is instantiated at the highly shifted eigenvalue $h̃(\hat{\lambda}_i)$. Because the highly discriminative filter is sensitive to frequency shifts, the response $h̃(\hat{\lambda}_i)$ can become very different from the intended response $h̃(\lambda_i)$.
3.  **Incompatibility Implied:** This inability to maintain sharp boundaries while handling perturbation means stability and discriminability are incompatible for graph filters. If a graph filter aims for stability (requiring it to be integral Lipschitz), it must be **flat at high frequencies**, which makes it impossible to discriminate high-frequency components. If it attempts high-frequency discrimination, it becomes **unstable**.

## Question 46 (2 points). We have signals $v_i$ and $v_j$ aligned with eigenvectors $i$ and $j$ of a shift operator. Explain how these signals can be discriminated with graph neural networks. What happens when the shift operator is perturbed? Your answer must imply that stability and discriminability can be compatible when we use graph neural networks.

#### Discrimination with Graph Neural Networks
If signals $v_i$ and $v_j$ correspond to high-frequency components, a GNN achieves stable discrimination by leveraging the cascade of integral Lipschitz filters and pointwise nonlinearities ($\sigma$).

1.  **Stable Layer 1 Isolation:** The GNN uses integral Lipschitz filters in Layer 1 to achieve stability, meaning the filter must be **flat** at the high frequencies $\lambda_i$ and $\lambda_j$. This filter does not discriminate between $v_i$ and $v_j$, but it **isolates** them from the rest of the spectrum.
2.  **Frequency Demodulation:** The pointwise nonlinearity $\sigma$ following Layer 1 acts as a **low-pass demodulator** (or frequency mixer). Applying $\sigma$ to the filter output ($\sigma(v_i)$ or $\sigma(v_j)$) spreads the energy across the entire frequency spectrum, including **low frequencies**.
3.  **Discrimination in Deeper Layers:** The energy components shifted to the low-frequency domain can then be **sharply discriminated** by the integral Lipschitz filter in Layer 2, because integral Lipschitz filters can be highly discriminative and stable at low frequencies.

#### Effect of Perturbation and Compatibility
When the shift operator $S$ is perturbed ($S \to \hat{S}$):

1.  **Stable Input to Nonlinearity:** Because the Layer 1 filter is integral Lipschitz, it is **stable** to the eigenvalue dilation. The resulting layer output $x^{(1)} = \sigma[H^1(\hat{S})x^{(0)}]$ remains **more or less the same** whether the filter is run on $S$ or $\hat{S}$.
2.  **Stable Demodulation:** Since the output of Layer 1 is stable, the distribution of energy into low frequencies is also stable.
3.  **Compatibility Implied:** GNNs demonstrate that stability and discriminability **can be compatible**. The GNN uses integral Lipschitz filters for **stability**, satisfying the prerequisite for stability to relative deformations. It achieves **discriminability** by transforming high-frequency features into low-frequency features using the pointwise nonlinearity $\sigma$, allowing subsequent filters to exploit the sharp, yet stable, discrimination properties available at low frequencies.

## Question 47 (2 points). We claim that stability analyses are key to explaining the improved performance of GNNs relative to graph filters. Explain.

The improved performance of GNNs relative to graph filters (GFs), despite their conceptual proximity, is explained by **stability analyses**.

1.  **GNN Performance Advantage:** GNNs often work significantly better than linear GFs in practice. This unexpected result warrants explanation.
2.  **The Tradeoff:** Stability analyses demonstrate that linear GFs suffer from a fundamental incompatibility: stability (required for meaningful perturbations) demands integral Lipschitz filters, which precludes high-frequency discriminability.
3.  **GNN Solution:** GNNs overcome this limitation by having a **better stability versus discriminability tradeoff**.
    *   GNNs maintain **stability** by inheriting the necessary properties of integral Lipschitz filters used in each layer.
    *   They gain **discriminability** because the pointwise nonlinearity ($\sigma$) demodulates unstable, high-frequency components of the signal into stable, low-frequency components.
    *   This allows the GNN to exploit **quasi-symmetries** of the graph more effectively than GFs. Since perfect symmetries are rare and quasi-symmetries are common, GNNs' superior ability to leverage quasi-symmetries explains their better empirical performance.

## Question 48 (2 points). When do you expect GNNs to have large advantages with respect to graph filters in solving machine learning problems? Are there machine learning problems where you expect GNNs to have marginal advantages with respect to graph filters?

GNNs outperform graph filters because they possess a superior stability-discriminability tradeoff.

#### Large Advantages Expected:
GNNs are expected to have a **large advantage** over GFs in problems characterized by:

1.  **High-Frequency Discrimination Needs:** When the features required for successful learning are represented by **high-frequency components** in the GFT domain. GFs must flatten their frequency response here to maintain stability, rendering them non-discriminative.
2.  **Meaningful Graph Perturbations/Quasi-Symmetries:** When the underlying graph support is subject to **meaningful deformations** (relative perturbations) or exhibits **quasi-symmetries**. In these scenarios, the inherent instability of discriminative GFs fails to generalize, whereas GNNs, through their nonlinearity, maintain both stability and discriminability.

#### Marginal Advantages Expected:
GNNs are expected to have **marginal advantages** over GFs in problems characterized by:

1.  **Low-Frequency Features:** When the learning problem is primarily a **low-frequency problem**, meaning the relevant information is contained in low-frequency GFT components. In this domain, integral Lipschitz filters (used by both GFs and GNNs) can be arbitrarily stable and discriminative.
2.  **Example (Recommendation Systems):** In recommendation systems, the customer similarity graph tends to be low-frequency. Experiments show that GNNs outperform graph filters, but only by a **small difference** (e.g., an extra 10% performance).

## Question 49 (2 points). Graph filters outperform linear regression in recommendation systems. The improved performance of graph filters is due to their ability to exploit permutation symmetries. Explain.

**Performance Comparison:** In recommendation systems (such as the MovieLens 100K dataset), graph filters significantly outperform linear regression. Linear regression's error (MSE $\approx 2$) is worse than the graph filter's error (MSE $\approx 1$). Importantly, the graph filter generalizes well to the test set, while linear regression performs even worse.

**Explanation via Permutation Symmetries:**

1.  **Architecture:** Both graph filters and linear regression rely on **linear** maps. Linear regression uses an arbitrary linear transformation $H$, while the graph filter uses a linear map restricted to a polynomial on the shift operator $S$ ($\sum h_k S^k$).
2.  **Leveraging Structure:** The improved performance of the graph filter is due to its ability to leverage the **underlying permutation symmetry** of the rating prediction problem.
3.  **Permutation Equivariance:** Graph filters are **permutation equivariant**. This property means that processing a permuted graph and signal results in a consistently permuted output.
4.  **Exploiting Symmetries:** This mathematical property allows the filter to exploit **internal symmetries** in graph signals. For instance, if two users have symmetric local rating patterns (a symmetry in the user graph), the filter learns how to process the ratings of both users by observing only one. This efficiently multiplies the available training data and explains the excellent generalization performance observed for the graph filter. Linear regression, which ignores the graph, fails to exploit these symmetries.

## Question 50 (2 points). Graph neural networks outperform fully connected neural networks in recommendation systems. The improved performance of GNNs is due to their ability to exploit permutation symmetries. Explain.

**Performance Comparison:** Graph Neural Networks (GNNs) outperform Fully Connected Neural Networks (FCNNs) in recommendation systems. Although FCNNs may achieve a lower loss on the training set because they are a superset of GNNs, this performance is **illusory** as FCNNs fail to generalize to unseen signals. GNNs generalize well, maintaining a stable test MSE.

**Explanation via Permutation Symmetries:**

1.  **Architecture:** FCNNs use arbitrary linear maps and pointwise nonlinearities, lacking knowledge of the underlying graph structure. GNNs impose the restriction that the linear maps must be **graph convolutional filters**, incorporating the graph shift operator $S$ as prior information.
2.  **Permutation Equivariance:** GNNs are **permutation equivariant**. This means that consistently relabeling the input signal $x$ and the graph shift operator $S$ results in a consistent relabeling of the output.
3.  **Exploiting Symmetries:** The equivariance property allows the GNN to successfully exploit the **internal symmetries** of the graph signals. When a graph has a symmetry (or quasi-symmetry), the GNN learns how to process a permuted input signal from having only observed the original signal, effectively leveraging the symmetry to multiply the size of the dataset and ensure good generalization. FCNNs, being unaware of the graph structure, cannot achieve this generalization.

## Question 51 (2 points). Graph neural networks outperform graph filters in recommendation systems. The improved performance of GNNs is due to their better stability vs discriminability tradeoff. Explain.

**GNN Performance in Recommendation Systems:**
In recommendation systems, experiments show that Graph Neural Networks (GNNs) outperform linear Graph Filters (GFs), although sometimes the difference is small (e.g., an extra 10% performance). The GNN generalizes well to the test set, while the linear GF also generalizes well, but the GNN achieves better performance.

**Explanation via Stability vs. Discriminability Tradeoff:**
The improved performance of the GNN is explained by its **superior stability versus discriminability tradeoff** compared to linear filters.

1.  **GF Limitation (Incompatibility):** Linear GFs must use filters that are **integral Lipschitz** to achieve stability against meaningful relative deformations of the graph. This condition forces the filter's frequency response to be **flat (non-discriminative) at high frequencies** ($\lambda$), meaning linear GFs **cannot** discriminate high-frequency features while remaining stable. Stability and discriminability are incompatible for GFs.
2.  **GNN Solution (Compatibility):** GNNs overcome this by layering integral Lipschitz filters with **pointwise nonlinearities ($\sigma$)**. The nonlinearity acts as a **low-pass demodulator**, converting unstable, high-frequency components of the signal into **stable, low-frequency components**. These low-frequency components can then be sharply discriminated by the stable integral Lipschitz filters in deeper layers.
3.  **Conclusion:** The GNN achieves a **legitimate tradeoff** where it is both stable (inherited from integral Lipschitz filters) and discriminative (due to the action of the nonlinearity). This superior capability to leverage quasi-symmetries explains why GNNs outperform GFs.

## Question 52 (2 points). In a recommendation system, the user similarity graph has adjacency matrix $S$ and user ratings are grouped in the vector $x$. The entries of $x$ that are not rated are zero. A common way of predicting ratings is the product $y = Sx$. Why is this a good idea?

Predicting ratings using the product $y = Sx$ is a simple form of a graph filter ($H(S)=S$) that leverages the inherent structure of the recommendation problem.

1.  **Diffusion and Locality:** Multiplication by the graph shift operator $S$ implements **diffusion of the signal over the graph**.
2.  **Weighted Averaging:** The $i$-th component of the resulting diffused signal $y$ is computed as $y_i = \sum_j w_{ij} x_j$. Here, $x$ is the vector of ratings, and $S$ is the **user similarity graph** where edges $w_{ij}$ express the expected similarity between components (ratings).
3.  **Leveraging Similarity:** When $y = Sx$ is computed, the predicted rating $y_i$ for user $i$ is a **weighted sum** of the ratings $x_j$ provided by their neighboring users $j$ (similar customers). **Stronger weights** (higher similarity scores) contribute more to the diffusion output.
4.  **Prediction Mechanism:** This operation leverages the graph structure to predict missing ratings by assuming that similar users will have similar tastes. In collaborative filtering, the graph informs the completion of ratings when some are unknown. This prediction method is used to eliminate large variability associated with unobserved ratings.

## Question 53 (2 points). In a distributed multiagent system, agents can communicate with nearby peers. This defines a communication graph $S$. A graph filter can be implemented in a way in which nodes exchange information over the communication graph. Explain.

A graph filter $H(S)$ is defined as a polynomial on the shift operator $S$: $y = H(S)x = \sum h_k S^k x$. This operation can be implemented in a distributed manner using the **recursive definition of the diffusion sequence**.

1.  **Shift Operator as Communication:** In a distributed multiagent system, the graph $S$ models the **locality of interactions** and restrictions on information exchange (physical proximity or communication limits).
2.  **Local Diffusion:** The multiplication of a signal $x^{(k)}$ by the shift operator $S$, $x^{(k+1)} = S x^{(k)}$, is a **local operation** where components are mixed only with those of **neighboring nodes**. This requires nodes to exchange information only with their immediate peers defined by the communication graph $S$.
3.  **Shift Register Implementation:** A graph convolution is equivalent to a shift register structure that performs: **Shift. Scale. Sum.**. In the context of a distributed system:
    *   **Shift:** Nodes locally exchange their current state $x^{(k)}$ with neighbors defined by $S$ to compute the next diffused signal $x^{(k+1)}$.
    *   **Scale and Sum:** Each node scales the diffused signal $S^k x$ by the filter coefficient $h_k$ and sums it to the accumulating output $y$.

Since the implementation relies only on **recursive application of the local shift operator $S$** and local scaling and summation, the graph filter can be implemented distributedly where agents exchange information solely over the communication graph $S$.

## Question 54 (2 points). A CNN is a particular case of a GNN. Explain.

A Convolutional Neural Network (CNN) is a **proper generalization** of a Graph Neural Network (GNN).

1.  **CNN Structure:** A CNN is made up of layers composing convolutional filters with pointwise nonlinearities.
2.  **Graph Interpretation:** Convolutions in time and space can be written as **polynomials on the adjacency matrices of specific underlying graphs**. Specifically, a convolution in time is a polynomial on the adjacency matrix of a **directed line graph**.
3.  **GNN Structure:** A GNN uses the same layered architecture but defines its linear maps as **graph convolutional filters**—polynomials on the shift operator $S$ of an **arbitrary graph**.
4.  **Particularization:** The GNN architecture recovers a CNN if the arbitrary shift operator $S$ is specialized to represent the **adjacency matrix of the directed line graph** (for time signals) or the grid graph (for images). Since the GNN encompasses this specialization, it is a **generalization** of the CNN.

## Question 55 (2 points). A GNN is a particular case of a fully connected NN. Explain.

A Graph Neural Network (GNN) is a particular case of a Fully Connected Neural Network (FCNN) because the linear maps within the GNN layers are restricted.

1.  **FCNN Architecture:** An FCNN uses a cascade of layers, each composing an **arbitrary linear map $H$** with a pointwise nonlinearity $\sigma$. The FCNN's optimization search space includes **all possible linear transformations** $H$.
2.  **GNN Architecture:** A GNN uses the same cascade structure, but the linear maps are restricted to be **graph convolutional filters** $H(S) = \sum h_k S^k$. This incorporates the graph structure ($S$) as prior information.
3.  **Restriction:** Since the graph convolutional filter is a specific, restricted linear map, the set of functions represented by a GNN is a **subset** of the functions represented by an FCNN.
4.  **Conclusion:** The GNN imposes a specific structure (a polynomial on $S$) onto the linear transformation $H$ used in the FCNN layer definition. Therefore, the GNN is a **particularization** of the fully connected NN.

## Question 56 (2 points). Convolutions in time are particular cases of convolutions in graphs. Explain.

This relationship is established by representing time signals using a specific graph structure.

1.  **Time Convolution Definition:** A time convolution is a weighted linear combination of **time-shifted versions** of the input signal.
2.  **Graph Representation of Time:** Discrete time can be described using a **directed line graph**, where the components of the time signal $x$ are associated with nodes.
3.  **Time Shift as Graph Shift:** Multiplication by the **adjacency matrix $S$ of the line graph** is equivalent to the time shift operation.
4.  **Convolution Equivalence:** Consequently, the time convolution can be mathematically rewritten as a **polynomial on the adjacency matrix $S$ of the line graph**: $y = \sum h_k S^k x$.
5.  **Graph Convolution Generalization:** A general graph convolution uses the exact same polynomial form $y = \sum h_k S^k x$, but $S$ is the shift operator of an **arbitrary graph**.
6.  **Conclusion:** The time convolution is recovered as a **particular case** of the graph convolution when the arbitrary shift operator $S$ is specified as the adjacency matrix of the line graph.

## Question 57 (2 points). Prove that a graph filter with an integral Lipschitz frequency response is stable to a scaling of the graph.

The theorem states that given shift operators $S$ and $\hat{S} = (1 + \varepsilon) S$, and an integral Lipschitz filter $H$ with constant $C$, the operator norm difference is bounded by $\|\mathbf{H}(\hat{S})-\mathbf{H}(S)\| \leq C \varepsilon + O(\varepsilon^2)$.

The proof involves demonstrating that the norm of the filter variation, $\Delta(S)$, is bounded by $C\varepsilon$ for unit norm inputs $x$:

1.  **Filter Variation:** The first-order filter variation is defined by the series expansion:
    $$\Delta(S) = H(\hat{S}) - H(S) \approx \varepsilon \sum_{k=0}^{\infty} k h_k S^k$$
2.  **GFT Domain Application:** Apply $\Delta(S)$ to a unit norm input signal $x$, written in the GFT basis ($x = \sum \tilde{x}_i v_i$):
    $$\Delta(S)x = \varepsilon \sum_{i=1}^n \tilde{x}_i \left( \sum_{k=0}^{\infty} k h_k \lambda_i^k \right) v_i$$
    Recognizing that $\sum k h_k \lambda_i^k = \lambda_i h'(\lambda_i)$ (the derivative of the frequency response scaled by $\lambda_i$):
    $$\Delta(S)x = \varepsilon \sum_{i=1}^n \tilde{x}_i \left( \lambda_i h'(\lambda_i) \right) v_i$$
3.  **Bounding the Norm:** Since the eigenvectors $v_i$ are orthogonal, the squared norm of the result is the sum of the squared components (Pythagoras's theorem):
    $$\|\Delta(S)x\|^2 = \varepsilon^2 \sum_{i=1}^n \tilde{x}_i^2 \left( \lambda_i h'(\lambda_i) \right)^2$$
4.  **Applying Integral Lipschitz Condition:** The **Integral Lipschitz hypothesis** states that $|\lambda h'(\lambda)| \leq C$. Applying this bound:
    $$\|\Delta(S)x\|^2 \leq \varepsilon^2 \sum_{i=1}^n \tilde{x}_i^2 C^2 = C^2 \varepsilon^2 \sum_{i=1}^n \tilde{x}_i^2$$
5.  **Conclusion:** Since the GFT is unitary, $\sum \tilde{x}_i^2 = \|x\|^2 = 1$. Taking the square root concludes the proof:
    $$\|\Delta(S)x\| \leq C \varepsilon$$

## Question 58 (2 points). Prove that a graph neural network with layers made up of integral Lipschitz filters is stable to a scaling of the graph. To prove this result it is convenient to begin by proving stability of graph filters.

The proof proceeds by induction, relying on the stability of the integral Lipschitz graph filters (proven in Q57/Source) and the properties of the nonlinearity.

The theorem states that given $\hat{S} = (1 + \varepsilon) S$ and a GNN $\Phi$ with $L$ layers using integral Lipschitz filters (constant $C$), the stability is bounded by $\|\Phi(\cdot; \hat{S}, \mathbf{H}) - \Phi(\cdot; S, \mathbf{H})\| \leq C L \varepsilon + O(\varepsilon^2)$.

1.  **Layer Output Difference:** Let $x^\ell$ be the output of Layer $\ell$ run on $S$, and $\hat{x}^\ell$ be the output run on $\hat{S}$. The layer operation is $x^\ell = \sigma[H^\ell(S) x^{\ell-1}]$.
2.  **Eliminating Nonlinearity:** The nonlinearity $\sigma$ is assumed to be **normalized Lipschitz** ($\|\sigma(x_2) - \sigma(x_1)\| \leq \|x_2 - x_1\|$). This allows the difference in layer outputs to be bounded by the difference in the linear filter outputs:
    $$\|\mathbf{x}^\ell - \hat{\mathbf{x}}^\ell\| \leq \|H^\ell(S)\mathbf{x}^{\ell-1} - H^\ell(\hat{S})\hat{\mathbf{x}}^{\ell-1}\|$$
3.  **Norm Manipulation (Triangle Inequality):** Using the identity $H^\ell(S)\mathbf{x}^{\ell-1} - H^\ell(\hat{S})\hat{\mathbf{x}}^{\ell-1} = [H^\ell(S) - H^\ell(\hat{S})]\mathbf{x}^{\ell-1} + H^\ell(\hat{S})[\mathbf{x}^{\ell-1} - \hat{\mathbf{x}}^{\ell-1}]$, we apply the triangle inequality and submultiplicative property:
    $$\|\mathbf{x}^\ell - \hat{\mathbf{x}}^\ell\| \leq \|H^\ell(S) - H^\ell(\hat{S})\| \|\mathbf{x}^{\ell-1}\| + \|H^\ell(\hat{S})\| \|\mathbf{x}^{\ell-1} - \hat{\mathbf{x}}^{\ell-1}\| + O(\varepsilon^2)$$
4.  **Applying Bounds (Inductive Step):** We use the normalization assumptions ($\|H^\ell(\hat{S})\|=1$, $\|\mathbf{x}^{\ell-1}\|\leq 1$). We then substitute the known **stability bound for the integral Lipschitz filter** (proven in Q57/Source): $\|H^\ell(S) - H^\ell(\hat{S})\| \leq C\varepsilon$:
    $$\|\mathbf{x}^\ell - \hat{\mathbf{x}}^\ell\| \leq C\varepsilon + \|\mathbf{x}^{\ell-1} - \hat{\mathbf{x}}^{\ell-1}\| + O(\varepsilon^2)$$
5.  **Accumulation:** Applying this recursion backward from Layer $L$ to Layer 1 (where the input distortion $\|\mathbf{x}^0 - \hat{\mathbf{x}}^0\|$ is zero) results in the linear accumulation of the distortion $C\varepsilon$ across $L$ layers, demonstrating that the **stability is inherited** from the filters and proves the final bound $C L \varepsilon$.
