## Question 1 (2 points)
A graphon $W$ is a function defined in $[0, 1]^2$ and taking values in $[0, 1]$. Explain how to use a graphon as a generative model of graphs with $n$ nodes. Illustrate this concept with a stochastic block model.

**Answer:**
A graphon $W$ is a bounded symmetric measurable function $W: [0, 1]^2 \to [0, 1]$. It serves as an abstraction for families of graphs with large numbers of nodes and similar structure. It can be used as a generative model for graphs through sampling.

**How to use a graphon as a generative model:**
1.  **Generate Vertices:** To create a graph $G_n$ with $n$ nodes, sample $n$ points $\{u_1, u_2, \ldots, u_n\}$ from the unit interval $[0, 1]$. These points can be chosen via regular partitions (e.g., on a grid) or sampled uniformly at random. Each sample $u_i$ corresponds to a node $i$ of the graph.
2.  **Generate Edges:** To determine the properties of the edge $(i, j)$, evaluate the graphon at the sampled points $W(u_i, u_j)$.
    *   **Stochastic Model:** An unweighted undirected edge is created between nodes $i$ and $j$ with probability $W(u_i, u_j)$.
    *   **Weighted Model:** A weighted undirected edge is created with weight equal to $W(u_i, u_j)$.

**Illustration with a Stochastic Block Model (SBM):**
A balanced stochastic block model (SBM) graphon is an example of a generative model that models community structure. This graphon takes two values, $p$ and $q$, based on where the arguments $u$ and $v$ fall within the unit square.

*   If $u$ and $v$ are both in $[0, 1/2]$ or both in $[1/2, 1]$ (representing two communities), $W(u, v)$ takes a value $p$ (strong connection).
*   If one argument is in $[0, 1/2]$ and the other is in $[1/2, 1]$, $W(u, v)$ takes a value $q$ (weak connection).

When generating a graph from this SBM graphon, nodes sampled in the first half of the interval are likely to connect strongly with each other (with probability $p$) but weakly with nodes sampled in the second half of the interval (with probability $q$), thus generating a graph that exhibits a balanced community structure.

---

## Question 2 (2 points)
We are given a graphon $W$ and an unweighted graph $G_n$ with $n$ nodes. Explain the concept of homomorphism density for the graphon $W$ and the unweighted graph $G_n$. Use these two definitions to define convergence of graph sequences $\{G_n\}^\infty_{n=1}$ to a graphon $W$ in the sense of homomorphism densities.

**Answer:**

**Homomorphism Density for an Unweighted Graph $G_n$:**
The homomorphism density $t(F, G_n)$ measures the relative frequency with which a small, finite, unweighted, and undirected graph $F$ (called a motif) appears in $G_n$.

A homomorphism $\beta$ is an adjacency preserving map from the vertices $V'$ of the motif $F=(V', E')$ into the vertices $V$ of the graph $G=(V, E)$, such that if $(i, j)$ is an edge in $E'$, then $(\beta(i), \beta(j))$ must be an edge in $E$.

If $G_n$ has $n$ nodes and the motif $F$ has $n'$ nodes, there are $n^{n'}$ possible maps from $F$ to $G_n$. The homomorphism density $t(F, G_n)$ is the fraction of these total maps that are homomorphisms:
$$t(F, G_n) = \frac{\text{hom}(F, G_n)}{n^{n'}}$$
where $\text{hom}(F, G_n)$ is the number of homomorphisms from $F$ to $G_n$.

**Homomorphism Density for a Graphon $W$:**
Homomorphisms are defined analogously for graphons. The homomorphism density $t(F, W)$ of a motif $F$ into the graphon $W$ is defined by an integral over the unit interval $[0, 1]^{n'}$ (where $n'$ is the number of nodes in $F$):
$$t(F, W) = \int_{[0, 1]^{n'}} \prod_{(i, j) \in E'} W(u_i, u_j) \prod_{i \in V'} du_i$$
This integral represents the probability of drawing the motif $F$ from the graphon $W$.

**Convergence in the Sense of Homomorphism Densities:**
A sequence of undirected graphs $\{G_n\}^\infty_{n=1}$ is said to converge to the graphon $W$ if, for all finite, unweighted, and undirected graphs $F$ (motifs), the homomorphism density of the graph sequence approaches the homomorphism density of the graphon:
$$\lim_{n \to \infty} t(F, G_n) = t(F, W)$$
If this condition holds, the sequence $G_n$ converges to $W$ in the homomorphism density sense.

---

## Question 3 (2 points)
Given a graphon $W(u, v)$ define a graphon signal $X(u)$ supported on this graphon. Introduce and explain the concept of a convergent sequence of graph signals.

**Answer:**

**Graphon Signal Definition:**
A graphon signal is defined as a pair $(W, X)$, where $W$ is the graphon and $X$ is a function $X: [0, 1] \to \mathbb{R}$. The function $X(u)$ is required to have finite energy, meaning $X$ belongs to the space $L_2([0, 1])$, such that $\int_0^1 |X(u)|^2 du < \infty$. Graphon signals serve as limit objects for convergent sequences of graph signals.

**Convergent Sequence of Graph Signals:**
A graph signal is a pair $(G_n, x_n)$, where $G_n$ is a graph and $x_n$ is a signal supported on its nodes. A sequence of graph signals $(G_n, x_n)$ is said to converge to the graphon signal $(W, X)$ if there exists a sequence of permutations $\{\pi_n\}$ such that, as $n \to \infty$:

1.  **Graph Convergence:** The graph sequence $G_n$ converges to the graphon $W$ in the homomorphism density sense for any simple graph $F$:
    $$\lim_{n \to \infty} t(F, G_n) \to t(F, W)$$
2.  **Signal Convergence:** The $L_2$ norm of the difference between the graphon signal $X$ and the induced graphon signal generated by the permuted graph signal $(\pi_n(G_n), \pi_n(x_n))$ converges to zero:
    $$\lim_{n \to \infty} \left\|X_{\pi_n(G_n)} - X\right\|_{L_2} = 0$$

**Explanation:**
*   **Permutation $(\pi_n)$:** The sequence of permutations $\{\pi_n\}$ is used to ensure the convergence definition is independent of the node labeling.
*   **Induced Graphon Signal ($X_{\pi_n(G_n)}$):** This concept is necessary because it is an "apples to apples" comparison: it transforms the vector signal $x_n$ (which has a finite number of components) into a function $X_{G_n}(u)$ supported on $[0, 1]$. This allows comparison with the limit graphon signal $X(u)$ using the $L_2$ norm. The induced signal $X_G(u)$ is constructed by assigning the value $x_i$ to the subinterval $I_i$ of a regular partition of $[0, 1]$.

---

## Question 4 (2 points)
Given a graphon $W(u, v)$ define a graphon shift $T_W$. Use this operator to define eigenvectors and eigenfunctions of a graphon $W$.

**Answer:**

**Graphon Shift Operator ($T_W$):**
The graphon $W(u, v)$ is used to define an integral linear operator $T_W$, also known as the Graphon Shift Operator (WSO), which maps signals in $L_2([0, 1])$ to signals in $L_2([0, 1])$.

When applied to a graphon signal $X(u)$, the resulting signal $(T_W X)(v)$ is defined by the integral:
$$(T_W X)(v) = \int_0^1 W(u, v) X(u) du$$
This operation is analogous to matrix multiplication in graph signal processing, resulting in the diffusion of the signal $X$ over the graphon $W$. Since $W$ is bounded and symmetric, $T_W$ is a self-adjoint Hilbert-Schmidt operator.

**Eigenfunctions and Eigenvalues of $W$:**
The eigenvalues ($\lambda$) and eigenfunctions ($\phi$) of the graphon $W$ are defined via the WSO $T_W$.

A function $\phi: [0, 1] \to \mathbb{R}$ is an eigenfunction of $T_W$ with associated eigenvalue $\lambda$ if, when the operator $T_W$ is applied to $\phi$, it results in a scaling of $\phi$ by $\lambda$:
$$(T_W \phi)(v) = \int_0^1 W(u, v) \phi(u) du = \lambda \phi(v)$$

The WSO $T_W$ has an infinite but countable number of eigenvalue-eigenfunction pairs $\{(\lambda_i, \phi_i)\}^\infty_{i=1}$. These eigenvalues are real and lie in the interval $[-1, 1]$. The eigenfunctions $\{\phi_i\}$ form an orthonormal basis of $L_2([0, 1])$.

---

## Question 5 (2 points)
Eigenvalues of a graph sequence $\{G_n\}^\infty_{n=1}$ that converges to a graphon $W(u, v)$ in the homomorphism density sense, converge to the graphon eigenvalues. Explain.

**Answer:**
This phenomenon is formally stated by the **Eigenvalue Convergence of a Graph Sequence Theorem**.

If a graph sequence $\{G_n\}$ converges to a graphon $W$ in the homomorphism density sense, the eigenvalues $\lambda_j(S_n)$ of the graph shift operator $S_n$, when normalized by $n$, converge to the eigenvalues $\lambda_j(T_W)$ of the graphon shift operator $T_W$ for all indices $j$:
$$\lim_{n \to \infty} \frac{\lambda_j(S_n)}{n} = \lambda_j(T_W)$$

This result is important because it confirms that the spectral properties of the sequence of graphs approach the spectral properties of the limiting graphon.

**Explanation:**
The convergence holds because the graph convergence (in the homomorphism density sense) is strongly linked to the spectrum of the induced operators.
The homomorphism density of a $k$-cycle motif, $t(C_k, W')$, is related to the sum of the $k$-th powers of the eigenvalues of the induced operator $T_{W'}$: $t(C_k, W') = \sum_{i \in \mathbb{Z}\setminus\{0\}} \lambda_i(T_{W'})^k$.

Since the graph sequence $G_n$ converges to $W$, the homomorphism densities must converge:
$$\lim_{n \to \infty} t(C_k, G_n) = t(C_k, W)$$

By linking the convergence of the homomorphism densities of cycles to the convergence of the sums of powers of normalized eigenvalues, it can be shown that the individual normalized graph eigenvalues converge to the corresponding graphon eigenvalues.

However, it is noted that while convergence holds for every index $j$, the index $n_0$ required to satisfy convergence to a given tolerance $\epsilon$ is different for each $j$, meaning the convergence of the eigenvalues is not uniform.

---

## Question 6 (2 points)
An eigenvector of a graph sequence that converges to a graphon converges to the graphon eigenvector. However, the convergence depends on how close the eigenvalue associated to the eigenvectors is to other eigenvalues. Explain.

**Answer:**
The convergence of an eigenvector (or eigenfunction) sequence is generally true when the graph sequence converges to the graphon. However, the rate of convergence is highly dependent on the separation of the associated eigenvalue from other eigenvalues, which is measured by the eigenvalue margin.

**Explanation:**
1.  **Eigenvalue Accumulation:** Graphon eigenvalues have the key property that they accumulate around $\lambda=0$. This means that for any small interval $[-c, c]$, there is an infinite number of eigenvalues clustered there.
2.  **Eigenvector Convergence Bound:** The convergence of the $j$-th eigenfunction $\phi_n^j$ of the induced graphon $W_{G_n}$ to the $j$-th eigenfunction $\phi_j$ of the graphon $W$ is characterized by a bound (derived from perturbation theory, such as the Davis-Kahan theorem):
    $$\left\|\phi_j - \phi_n^j\right\| \leq \frac{\pi}{2} \frac{\|W - W_{G_n}\|}{\text{d}(\lambda_j, \lambda_n^j)}$$
3.  **Dependence on Eigenvalue Margin:** The term $\text{d}(\lambda_j, \lambda_n^j)$ is the eigenvalue margin, which quantifies the distance of the eigenvalue pair $(\lambda_j, \lambda_n^j)$ to all other nearby eigenvalues. The convergence bound shows an inverse relationship: as the eigenvalue margin $d$ decreases, the distance between the corresponding eigenfunctions increases relative to the error $\|W - W_{G_n}\|$.
4.  **Slow Convergence near Zero:** For eigenvalues far from zero, the margin $d$ is large, and convergence is ready. However, near $\lambda=0$, where eigenvalues accumulate, the margins become very small or vanish. When the margin vanishes, the bound fails to ensure convergence for a fixed $n$ and arbitrary error $\epsilon$, meaning eigenvectors are slow to converge. This challenge precludes claiming uniform convergence of the full set of eigenvectors.

The convergence of eigenvectors is thus highly conditional on avoiding regions of the spectrum where eigenvalues cluster closely.

---

## Question 7 (2 points)
Given a graphon $W(u, v)$ we define a graphon shift $T_W$. Define a graphon filter to process signals $X(u)$ supported on this graphon. Graphon filters have the same algebraic structure of graph filters. Explain.

**Answer:**

**Definition of a Graphon Filter:**
The graphon $W(u, v)$ defines the Graphon Shift Operator (WSO), denoted $T_W$, which is an integral linear operator applied to a graphon signal $X(u) \in L_2([0, 1])$. The WSO calculates the diffused signal $(T_W X)(v)$ as:
$$(T_W X)(v) = \int_0^1 W(u, v) X(u) du$$

By recursively applying the WSO, we generate the graphon diffusion sequence. A graphon filter $T_H$ of order $K$ (with coefficients $h_k$) uses this sequence to process the input signal $X$ and produce the output signal $Y$. The filter is defined as a linear combination of these diffused signals:
$$Y(v) = (T_H X)(v) = \sum_{k=0}^K h_k (T_W^{(k)} X)(v)$$
where $T_W^{(k)}$ represents $k$ applications of $T_W$, and $T_W^{(0)} X = X$.

**Algebraic Structure Equivalence:**
A graphon filter possesses the same algebraic structure as a graph filter.
*   **Graph Filters** are defined as polynomials on the Graph Shift Operator (GSO), $S_n$.
*   **Graphon Filters** are defined as polynomials on the Graphon Shift Operator (WSO), $T_W$.

The underlying reason for this shared structure stems from **Algebraic Signal Processing (ASP)**. Both GSP and Graphon SP (WSP) are particular cases of ASP where the defining algebra is the algebra of polynomials of a single variable, $t$. The filters $H_n$ and $T_H$ are instantiated from the same abstract polynomial, $\sum h_k t^k$, but by mapping the generator $t$ to different shift operators $\rho(t)$:
*   For GSP, $\rho(t) = S_n$ (the adjacency matrix).
*   For WSP, $\rho(t) = T_W$ (the integral operator).

Because the algebraic structure of the polynomial (summation and product/composition) is preserved by the homomorphism $\rho$, both filters are constructed using the same sequence of shift, scale, and sum operations.

---

## Question 8 (2 points)
Consider sequences of graphs and graph signals converging to corresponding graphons and graphon signals. For a given set of filter coefficients define: (1) A sequence of graph filters. (2) A graphon filter. (3) The graph and graphon filter frequency response. (4) The graph filter frequency representation. (5) The graphon filter frequency representation. They are all the same polynomial. Explain.

**Answer:**
Given a set of filter coefficients $\{h_k\}$, the five defined concepts are all instantiations or representations derived from the same underlying abstract polynomial.

| Definition | Variable | Expression |
| :--- | :--- | :--- |
| **(1) Sequence of Graph Filters ($H_n$)**<br>The operational filter sequence applied to graph signals $x_n$. | Graph Shift Operator $S_n$ | $H_n(S_n) = \sum_{k=1}^K h_k S_n^k$ |
| **(2) Graphon Filter ($T_H$)**<br>The operational filter applied to the graphon signal $X$. | Graphon Shift Operator $T_W$ | $T_H = \sum_{k=1}^K h_k T_W^{(k)}$ |
| **(3) Frequency Response ($h(\lambda)$)**<br>The abstract filter function, identical for both graph and graphon filters. | Scalar variable $\lambda$ | $h(\lambda) = \sum_{k=0}^K h_k \lambda^k$ |
| **(4) Graph Filter Frequency Representation ($\hat{H}_n$)**<br>The frequency response evaluated at the normalized graph eigenvalues $\lambda_{n,j}$. | Normalized Eigenvalue $\lambda_{n,j}$ | $\hat{H}_n(\lambda_{n,j}) = \sum_{k=1}^K h_k \lambda_{n,j}^k$ |
| **(5) Graphon Filter Frequency Representation ($\hat{T}_H$)**<br>The frequency response evaluated at the graphon eigenvalues $\lambda_j$. | Graphon Eigenvalue $\lambda_j$ | $\hat{T}_H(\lambda_j) = \sum_{k=1}^K h_k \lambda_j^k$ |

**Explanation of Equivalence:**
The core reason they share the same form is that they are based on the same filter coefficients $\{h_k\}$.
*   **Node Domain Polynomials (1 & 2):** $H_n(S_n)$ and $T_H$ are the instantiation of the filter polynomial (P2) using the algebraic operations of their respective operator spaces (matrices for $S_n$, linear functionals for $T_W$).
*   **Spectral Domain Polynomials (3, 4, & 5):** The frequency response $h(\lambda)$ is the simplest polynomial form (P3), defined over the field operations (scalar product and sum). Since the abstract filter (P1) is the same, the frequency response derived from it is identical for all instantiations (graph and graphon). The frequency representations (4 and 5) are simply the evaluation of the frequency response polynomial $h(\lambda)$ at the eigenvalues corresponding to $S_n$ or $T_W$.

Because they all map back to the same set of coefficients $\{h_k\}$, they exhibit the same polynomial structure.

---

## Question 9 (2 points)
Consider sequences of graphs and graph signals converging to corresponding graphons and graphon signals. For a given set of filter coefficients define: (1) The graph and graphon filter frequency response. (2) The graph filter frequency representation. (3) The graphon filter frequency representation. Prove that the frequency representations of the graph filter sequence converges to the frequency representation of the graphon filter if the eigenvalues of the graph sequence converge to the eigenvalues of the graphon.

**Answer:**
We prove the convergence of the frequency representations, $\hat{H}_n(\lambda_{n,j}) \to \hat{T}_H(\lambda_j)$, relying on the convergence of the normalized eigenvalues $\lambda_{n,j} \to \lambda_j$.

**Definitions:**
*   **Frequency Response ($h(\lambda)$):** $h(\lambda) = \sum_{k=0}^K h_k \lambda^k$. This is a continuous polynomial function.
*   **Graph Frequency Representation ($\hat{H}_n(\lambda_{n,j})$):** $h(\lambda_{n,j})$ where $\lambda_{n,j} = \lambda_j(S_n)/n$.
*   **Graphon Frequency Representation ($\hat{T}_H(\lambda_j)$):** $h(\lambda_j)$ where $\lambda_j = \lambda_j(T_W)$.

**Proof of Convergence (Theorem 1):**
The convergence of the frequency representations is established by relating the convergence of the spectral variables to the continuity of the filter response.

1.  **Hypothesis (Eigenvalue Convergence):** The prerequisite for this proof is the Eigenvalue Convergence Theorem, which states that for a graph sequence $\{G_n\}$ converging to $W$ in the homomorphism density sense, the normalized graph eigenvalues converge to the graphon eigenvalues for all indices $j$:
    $$\lim_{n \to \infty} \lambda_{n,j} = \lim_{n \to \infty} \frac{\lambda_j(S_n)}{n} = \lambda_j(T_W)$$
2.  **Continuity of $h(\lambda)$:** Since the frequency response $h(\lambda)$ is defined as a polynomial, $h(\lambda) = \sum_{k=0}^K h_k \lambda^k$, it is inherently a continuous function.
3.  **Conclusion by Continuity:** By applying the continuity property of $h(\lambda)$: if the input to a continuous function converges, the output must also converge. Therefore, as $n \to \infty$:
    $$\lim_{n \to \infty} \hat{H}_n(\lambda_{n,j}) = \lim_{n \to \infty} h(\lambda_{n,j}) = h\left(\lim_{n \to \infty} \lambda_{n,j}\right) = h(\lambda_j) = \hat{T}_H(\lambda_j)$$

This proves that the frequency representation of the graph filter sequence converges to the frequency representation of the graphon filter.

---

## Question 10 (2 points)
Consider sequences of graphs and graph signals converging to corresponding graphons and graphon signals. For a given set of filter coefficients define: (1) A sequence of graph filters. (2) A graphon filter. (3) The graph and graphon filter frequency response. (4) The graph filter frequency representation. (5) The graphon filter frequency representation. Convergence of graph filter outputs to graphon filter outputs does not follow from convergence of frequency representations. Explain why.

**Answer:**
Although the filter frequency representations converge (as shown in Question 9), this does not guarantee that the output graph signals $y_n = H_n(S_n) x_n$ converge to the output graphon signal $Y = T_H X$ in the node domain.

**Explanation:**
To transition from the frequency domain back to the node domain, the Inverse Graph Fourier Transform (iGFT) and the Inverse Graphon Fourier Transform (iWFT) must be used. Convergence in the node domain requires the convergence of the entire sequence of operations:
$$y_n = \text{iGFT}_n \{ \hat{H}_n \cdot \hat{x}_n \} \quad \longrightarrow \quad Y = \text{iWFT} \{ \hat{T}_H \cdot \hat{X} \}$$

The convergence of the filter outputs $(G_n, y_n) \to (W, Y)$ fails in general because the inverse Fourier transforms (iGFT and iWFT) themselves do not converge uniformly.

The reasons for this failure relate to the underlying basis of the transform: the eigenvectors and eigenfunctions.
1.  **Dependence on Eigenvectors:** The GFT/WFT are defined as projections of the signal onto the eigenvectors ($\mathbf{v}_{n,j}$) or eigenfunctions ($\phi_j$).
2.  **Non-uniform Eigenvector Convergence:** While the eigenvalues converge (Question 9), the associated eigenvectors/eigenfunctions do not converge uniformly across all indices $j$.
3.  **Accumulation at Zero:** Graphon eigenvalues accumulate at $\lambda=0$. As $n \to \infty$, the corresponding eigenvalue margins for eigenvectors near zero vanish. This vanishing margin means that the distance between the graph eigenvectors and graphon eigenfunctions is not sufficiently bounded near zero to ensure convergence for all $j$.

Since the transformation used to map spectral representations back to node-domain signals ($\text{iGFT} \to \text{iWFT}$) is unstable or non-convergent near $\lambda=0$, the convergence of the filter outputs cannot be guaranteed merely by the convergence of the frequency representations.

*To visualize this relationship, think of the filter outputs as a movie composed frame-by-frame (the frequency representation polynomial) projected onto a screen (the eigenvector basis). The frequency representation (the polynomial) is stable and converges, like the plot points in the script. However, the projection system (the eigenvectors) is faulty, especially where the plot points cluster tightly ($\lambda \approx 0$). Although the script is correct, the resulting film quality (the filter output) cannot be guaranteed to converge perfectly because the unreliable projection smears the clustered parts of the image.*

---

## Question 11 (2 points)
Convergence of graph filter outputs to graphon filter outputs is possible provided that the filters have Lipschitz frequency responses. Explain why. State a theorem claiming convergence and discuss its implications.

**Answer:**

**Explanation of Convergence via Lipschitz Continuity:**
Convergence of graph filter outputs to graphon filter outputs ($\{(G_n, y_n)\} \to (W, Y)$) cannot generally be proven using the convergence of frequency representations alone because the inverse Graph Fourier Transform (iGFT) does not converge to the inverse Graphon Fourier Transform (iWFT) due to the non-uniform convergence of eigenvectors near $\lambda=0$.

However, imposing the constraint that the filter frequency response $h(\lambda)$ is Lipschitz continuous ensures convergence in the node domain. A function $h(\lambda)$ is L-Lipschitz continuous if, for all $\lambda, \lambda' \in \mathbb{R}$, the bound $|\mathbf{h}(\lambda) - \mathbf{h}(\lambda')| \leq L|\lambda - \lambda'|$ holds.

The value of this condition is that it bounds the filter's rate of variability. The challenge to eigenvector convergence arises from the accumulation of graphon eigenvalues around $\lambda=0$. When the filter is Lipschitz, the continuity hypothesis ensures that spectral components associated with eigenvalues clustered near zero are multiplied by similar numbers. Therefore, even if the eigenvectors cannot be perfectly distinguished (i.e., convergence is not uniform), the filter itself does not distinguish among these components either, allowing the convergence proof to hold regardless of the complexity introduced by the clustering eigenvalues.

**Theorem Claiming Convergence:**
The general convergence conclusion for Lipschitz continuous graph filters and arbitrary graphons is given by the following theorem:

**Theorem 3 (Convergence of filter response for Lipschitz continuous graph filters):** Let $\{(G_n, y_n)\}$ be the sequence of graph signals obtained by applying filters $H_n(S_n)$ to the sequence $\{(G_n, x_n)\}$, and let $(W, Y)$ be the graphon signal obtained by applying the graphon filter $T_H X$ to the signal $(W, X)$. If $\{(G_n, x_n)\}$ converges to $(W, X)$ and the function $h$ is L-Lipschitz, then $\{(G_n, y_n)\}$ converges to $(W, Y)$.

**Implications:**
The convergence of the outputs when $h(\lambda)$ is Lipschitz provides an initial approach to the transferability of graph filters. However, this result identifies a fundamental issue: transferability is counter to discriminability. A filter that is Lipschitz converges quickly to the graphon limit and transfers well, but because its rate of variability is limited by the Lipschitz constant $L$, it cannot discriminate between components associated with eigenvalues close to $\lambda=0$.

---

## Question 12 (2 points)
Graph filters that are more discriminative are more difficult to transfer to larger graphs.

**Answer:**
This statement reflects the transferability-discriminability non-tradeoff observed in graph filters.

**Explanation:**
1.  A filter's discriminability is related to the sharpness of its frequency response, which is quantified by the filter's Lipschitz constant ($L_2$). A larger $L_2$ means the filter is sharper and thus more discriminative, particularly near $\lambda=0$ where eigenvalues accumulate.
2.  However, the quality of the transfer (i.e., how close two filter outputs $Y_n$ and $Y_m$ on different graphs $G_n$ and $G_m$ are) is quantified by an approximation bound. This bound explicitly depends on $L_2$. According to the graph filter transferability theorem, the distance between the two induced graphon signals $\|Y_n - Y_m\|$ is bounded by a term that is linearly proportional to the filter's Lipschitz constant $L_2$.
3.  If the filter is made more discriminative (increasing $L_2$), the approximation bound grows. This increased bound means that the convergence guarantee is weakened, and the filter becomes more difficult to transfer between graphs. This relationship is true for both the spectral components that are easy to transfer ($|\lambda| > c$) and those that are difficult to transfer ($|\lambda| \leq c$).

---

## Question 13 (2 points)
Transferability to larger graphs and discriminability are incompatible in graph filters. Explain

**Answer:**
The relationship between transferability and discriminability in graph filters is described as an incompatibility or non-tradeoff.

**Explanation:**
The core reason for this incompatibility is the spectral property of graphons: the eigenvalues accumulate at $\lambda=0$.

*   **Discriminability Requirement:** To effectively discriminate components associated with eigenvalues clustered around $\lambda=0$, the filter must be sharp, meaning its frequency response must be highly variable in that range. This requires a large Lipschitz constant $L_2$.
*   **Transferability Requirement:** To ensure convergence or tight approximation bounds (transferability), the filter must possess low variability, which requires a small Lipschitz constant $L_2$.
*   **Incompatibility:** If the filter is designed to be highly discriminative (large $L_2$), the transferability bound becomes too large, rendering the bound "useless" for claiming transferability between graphs.

Consequently, graph filters face a dilemma: they must either be wide filters (transferable but not discriminative) or thin filters (discriminative but not transferable, as they take too long to converge to the graphon limit). The requirement of using graph filters (polynomials on the shift operator) prevents resolving this inherent tension.

---

## Question 14 (2 points)
Define a graphon neural network (WNN). WNNs do not exist in reality. Explain. Why then, do we define WNNs?

**Answer:**

**Definition of a Graphon Neural Network (WNN):**
A Graphon Neural Network (WNN) is a layered architecture designed to process graphon signals $(W, X)$. The architecture is defined by a composition of operations where each layer $l$ applies a graphon convolution (a polynomial of the Graphon Shift Operator $T_W$) followed by a pointwise nonlinearity $\sigma$.

The operation at layer $l$ is defined as:
$$X_l^f = \sigma \left(\sum_{g=1}^{F_{l-1}} h_{kl}^{fg} T_W^{(k)} X_l^{g-1}\right)$$
where $H = \{h_{kl}^{fg}\}$ groups the learnable parameters, and $Y = X_L$ is the final output.

**Why WNNs Do Not Exist in Reality:**
A graphon $W$ is defined as a bounded symmetric measurable function $W: [0, 1]^2 \to [0, 1]$, which serves as an abstraction for graphs with an uncountable number of nodes. Since real-world graphs and practical computing systems operate on a finite, countable number of nodes, WNNs, which operate on this limiting, infinite structure, do not exist in reality.

**Purpose of Defining WNNs:**
WNNs are defined primarily because they serve as limit objects and generative models for Graph Neural Networks (GNNs).
1.  **Generative Model:** The WNN structure $\Phi(H; W; X)$ shares the same filter parameters $H$ as the GNN structure $\Phi(H; S_n; x_n)$, regardless of the graphon $W$. This means that WNNs can be viewed as generative models for GNNs, enabling GNNs to be built as instantiations of the WNN.
2.  **Transferability and Scalability Analysis:** By defining WNNs, researchers can use them as a common abstract platform to analyze fundamental properties of GNNs, such as stability and, critically, transferability, in the limit as the number of nodes $n \to \infty$. This allows for the characterization of how GNN properties behave when the graph structure changes or scales.

---

## Question 15 (2 points)
Use graphon neural network (WNNs) to explain that graph neural networks (GNNs) inherit the transferability properties of graph filters.

**Answer:**
GNNs inherit the transferability properties of graph filters because the WNN provides a common reference limit object that bounds the distance between the outputs of GNNs running on different finite graphs.

**Explanation using WNNs:**
1.  **WNN as a Bridge:** A Graphon Neural Network ($Y = \Phi(H; W, X)$) is defined as the limit object for a sequence of GNNs. We can sample two different graph signals, $(S_n, x_n)$ and $(S_m, x_m)$, from the same graphon signal $(W, X)$. We then run GNNs with the same coefficients $H$ on both graphs, yielding outputs $y_n$ and $y_m$.
2.  **Approximation Theorem (GNN $\to$ WNN):** The GNN-WNN approximation theorem bounds the error between the induced graphon output of the GNN ($Y_n$) and the true WNN output ($Y$). This bound confirms that $Y_n$ is close to $Y$. Crucially, this approximation bound is an extrapolation of the graph filter approximation bound (Theorem 1, Question 13), scaled by the GNN depth ($L$) and width ($F$). This link shows that the convergence property of GNNs stems directly from the convergence property of their constituent graph filters.
3.  **Transferability via Triangle Inequality:** To prove that GNNs are transferable (i.e., $Y_n$ is close to $Y_m$), we use the WNN output $Y$ as an intermediary in the triangle inequality:
    $$\|Y_n - Y_m\|_{L_2} \leq \|Y_n - Y\|_{L_2} + \|Y - Y_m\|_{L_2}$$
    Since both terms on the right-hand side (the GNN-WNN approximation error bounds) vanish as the number of nodes $n$ and $m$ grow, it follows that the distance between the two GNN outputs must also vanish.

In essence, because the GNN inherits the filter's ability to approximate the continuous WNN, the WNN acts as a common reference point, ensuring that any two GNNs sampled from the same family (graphon) are close to each other, thereby inheriting the property of transferability initially analyzed for simple graph filters.

---

## Question 16 (2 points)
Define a Markov stochastic process and explain why in this case, learning from a sequence is equivalent to a sequence of learning problems.

**Answer:**

**Definition of a Markov Stochastic Process:**
A stochastic process (or random sequence) $x_t$ is defined as a Markov or memoryless process if the conditional probability of observing the next state $x_{t+1}$, given the complete history of the process $x_{1:t}$, is equal to the conditional probability given only the current state $x_t$.

Formally:
$$p \left( x_{t+1} \mid x_{1:t} \right) = p \left( x_{t+1} \mid x_t \right)$$

This definition implies that the future state, given the present state, is independent of the past. If the process also has outputs $y_t$, these outputs are conditionally independent; the probability of $y_t$ depends only on the current state $x_t$.

**Learning Equivalence:**
In a memoryless Markov process, learning is equivalent to a sequence of learning problems.
1.  **Memoryless Transitions:** The state evolution is a chain of memoryless transitions, meaning the transition from $x_t$ to $x_{t+1}$ depends only on $x_t$.
2.  **Output Dependence:** The outputs $y_t$ depend solely on the current state $x_t$.
3.  **AI Focus:** An AI designed to predict the output $\hat{y}_t$ only needs to mimic the conditional distribution of the observations $y_t$ given the present state $x_t$. Since the past is irrelevant for predicting the future in a Markov process, the AI does not need to consider the history of the sequence when making predictions.

---

## Question 17 (2 points)
Define a hidden Markov model. and explain why in this case learning from a sequence is not the same as a sequence of learning problems.

**Answer:**

**Definition of a Hidden Markov Model (HMM):**
A stochastic process $x_t$ follows a Hidden Markov Model (HMM) if there exists an auxiliary process $z_t$ (called the hidden state) such that:
1.  The hidden state $z_t$ is a memoryless Markov stochastic process: $p(z_{t+1} \mid z_{1:t}) = p(z_{t+1} \mid z_t)$.
2.  The observed state $x_t$ is conditionally independent, depending only on the current hidden state $z_t$: $p(x_t \mid z_t) = p(x_t \mid z_{0:t})$.
3.  Outputs $y_t$ are conditionally independent when given the hidden state $z_t$.

The key distinction is that the hidden state $z_t$ is unobservable.

**Why Learning from a Sequence is Necessary:**
In a hidden Markov model, learning is not equivalent to a sequence of learning problems:
1.  **AI Goal vs. Access:** To predict the outputs, the AI ideally needs to mimic the conditional distribution of $y_t$ given the hidden state $z_t$. However, since $z_t$ is hidden, the AI does not have access to it.
2.  **Insufficient Observation:** What the AI does observe is $x_t$, but this observed state alone is not sufficient to neglect the history of the process. The system state might be Markov, but only on the full hidden state (e.g., position, velocity, acceleration), and since only the observed state (e.g., position) is available, the past is relevant for prediction.
3.  **Need for Memory:** The process must be modeled as learning from a sequence because the output prediction at time $t$ depends on the entire history of observations $x_{1:t}$.

---

## Question 18 (2 points)
Define a recurrent neural network (RNN) and explain how it is well suited to the processing of a hidden Markov sequence $x_t$.

**Answer:**

**Definition of a Recurrent Neural Network (RNN):**
A Recurrent Neural Network (RNN) is defined by two separate learning parameterizations:
1.  **Hidden State Update ($\Phi_1$):** This maps the observed state $x_t$ and the previous hidden state $z_{t-1}$ to the current hidden state $z_t$. This is typically implemented as a perceptron: $z_t = \sigma(A x_t + B z_{t-1})$.
2.  **Output Prediction ($\Phi_2$):** This maps the updated hidden state $z_t$ to the predicted output $\hat{y}_t$.

The architecture is called recurrent because the hidden state $z_t$ is fed back as an input to calculate the hidden state for the next time step. The number of learnable parameters (the entries of matrices $A, B, C$) does not depend on the time index $t$, which allows for processing sequences of variable length.

**Suitability for Processing Hidden Markov Sequences:**
The RNN architecture is specifically designed to handle the dependency on history present in non-Markovian sequences like HMMs:
1.  **Hidden State Estimation:** RNNs utilize the observations of the observable state $x_t$ to estimate the hidden state $z_t$.
2.  **Encoding Past Information:** The recurrence mechanism allows the RNN to encode past information received from the data points seen so far into the hidden state $z_t$.
3.  **Circumventing Memory Growth:** By repeatedly updating the hidden state with each new data sample, the RNN implicitly creates a mapping from the history of the process to the current hidden state without having to store and process the entire history, thus circumventing the unbounded memory growth challenge.

---

## Question 19 (2 points)
Define a graph recurrent neural network (GRNN) and explain how it is well suited to the processing of a hidden Markov sequence $x_t$ whose components at each time $t$ are graph signals.

**Answer:**

**Definition of a Graph Recurrent Neural Network (GRNN):**
A Graph Recurrent Neural Network (GRNN) is a particular case of an RNN defined for a time-varying process $x_t$ where the signals observed at each time step $t$ are graph signals supported on a common graph shift operator (GSO) $S$. A GRNN combines elements of a GNN and an RNN.

The GRNN structure requires that the linear operations (which use matrices $A, B, C$ in a standard RNN) are replaced by graph filters (polynomials on the GSO $S$):
1.  **Hidden State Update:** The updated hidden state $z_t$ is a graph signal resulting from a perceptron that uses graph filters $A(S)$ and $B(S)$:
    $$z_t = \sigma \left( A(S)x_t + B(S)z_{t-1} \right)$$
    where $A(S)$ and $B(S)$ are polynomials on $S$ (e.g., $\sum_{k=0}^{K-1} a_k S^k$).
2.  **Output Prediction:** The output estimate $\hat{y}_t$ is calculated using a graph filter $C(S)$ applied to the hidden state $z_t$:
    $$\hat{y}_t = \rho \left( C(S)z_t \right)$$

**Suitability for Graph Signals in an HMM:**
The GRNN is well suited for processing a hidden Markov sequence $x_t$ where components are graph signals because it exploits both the temporal and spatial structure of the data:
1.  **Temporal Processing (RNN Inheritance):** It retains the recurrent structure necessary to estimate the hidden state $z_t$ from the history of the sequence, addressing the fundamental challenge posed by the hidden nature of the HMM.
2.  **Spatial Processing (GNN Integration):** By using graph filters $A(S), B(S), C(S)$, and ensuring that the hidden state $z_t$ is also a graph signal, the GRNN leverages the underlying graph structure of the data $x_t$. This approach ensures the architecture inherits desirable properties of GNNs, such as stability and permutation equivariance, making the learning scalable.

---

## Question 20 (2 points)
Explain the problem of vanishing gradients in RNNs.

**Answer:**
The problem of vanishing gradients occurs in RNNs during training via backpropagation through time when the network attempts to model long-term dependencies of length $T$.

**Explanation:**
1.  **Gradient Dependence on Time:** The calculation of the gradient (e.g., the Jacobian $\partial z_T / \partial B$ of the hidden state $z_T$ with respect to the weight matrix $B$) involves a product chain due to the recurrence relationship. If the simple state update $z_t = B z_{t-1}$ is considered (omitting input $x_t$ and nonlinearity $\sigma$), the hidden state $z_T$ depends on $B$ raised to the power of $T$: $z_T = B^T z_{t-T}$.
2.  **Eigenvalue Effect:** If the matrix $B$ is decomposed by its eigenvalues ($\Lambda$), the recurrence involves $\Lambda^T$.
3.  **Vanishing:** If the eigenvalues of $B$ are less than 1, raising them to the power of $T$ causes them to rapidly vanish (tend toward zero). This results in exponentially smaller gradients, making it difficult for the network to learn weights associated with long-term relationships.
4.  **Exploding:** Conversely, if the eigenvalues of $B$ are larger than 1, the gradients explode, leading to exponentially larger weights.
5.  **Loss of Information:** This phenomenon means that any component of the initial state $z_{t-T}$ that is not strongly aligned with the largest eigenvalues of $B$ will be amplified or, more often, discarded as $T$ increases. Consequently, the RNN fails to encode and utilize information from the distant past.

---

## Question 21 (2 points)
Define algebraic signal processing models and discuss how graph signal processing follows as a particular case.

**Answer:**
An Algebraic Signal Processing (ASP) model is fundamentally defined by a triplet $(A, M, \rho)$:
1.  **Algebra ($A$):** This is an associative algebra with unity that defines the set of allowable filters $h$. It prescribes the abstract rules of convolutional signal processing.
2.  **Vector Space ($M$):** This is the space where the signals $x$ live and are processed (e.g., vectors, functions, or sequences).
3.  **Homomorphism ($\rho$):** This is an operation-preserving map $\rho: A \to \text{End}(M)$, where $\text{End}(M)$ is the algebra of endomorphisms (linear transformations) of $M$. The homomorphism instantiates the abstract filter $h \in A$ into a concrete linear transformation $\rho(h)$ that can be applied to the signal $x \in M$.

**Graph Signal Processing (GSP) as a Particular Case:**
Graph Signal Processing (GSP) is recovered as a particular case of ASP by specifying the three components based on a graph with $n$ nodes and a shift operator $S$:

| ASP Component | Specification for GSP | Explanation |
| :--- | :--- | :--- |
| **Vector Space ($M$)** | $\mathbb{R}^n$ | Signals $x$ are vectors with $n$ components, representing node features. |
| **Algebra ($A$)** | The algebra of polynomials $P(t)$ | Filters $h$ are abstract symbolic polynomials $h = \sum_k h_k t^k$. |
| **Homomorphism ($\rho$)** | Defined by mapping the generator $t$ to the Graph Shift Operator $S$: $\rho(t) = S$. | The filter $h$ is instantiated as a polynomial on $S$: $\rho(h) = \sum_k h_k S^k$. |

---

## Question 22 (2 points)
Define algebraic Signal processing model and discuss how discrete time signal processing with 1D convolutional filters follows as a particular case.

**Answer:**
The definition of the ASP model is provided in the answer to Question 21.

**Discrete Time Signal Processing (DTSP) as a Particular Case:**
Discrete Time Signal Processing (DTSP) with 1D convolutional filters follows as a particular case of ASP by specifying the components based on infinite-length, discrete-time sequences:

| ASP Component | Specification for DTSP | Explanation |
| :--- | :--- | :--- |
| **Vector Space ($M$)** | $L_2(\mathbb{Z})$ | Signals $X$ are sequences defined over the integers with finite energy (square summable). |
| **Algebra ($A$)** | The algebra of polynomials $P(t)$ | Filters are abstract polynomials $h = \sum_k h_k t^k$, the same algebra used in GSP. |
| **Homomorphism ($\rho$)** | Defined by mapping the generator $t$ to the time shift operator $S$: $\rho(t) = S$. | The operator $S$ shifts time indices: $(SX)_n = (X)_{n-1}$. The resulting filter action, when applied to $X$, yields the familiar time convolution $(Y)_n = \sum_k h_k (X)_{n-k}$. |

---

## Question 23 (2 points)
Define algebraic Signal processing model and discuss how image processing with 2D convolutional filters follows as a particular case.

**Answer:**
The definition of the ASP model is provided in the answer to Question 21.

**Image Processing (IP) as a Particular Case:**
Image Processing (IP) with 2D convolutional filters follows as a particular case of ASP by dealing with signals indexed by two spatial coordinates:

| ASP Component | Specification for IP | Explanation |
| :--- | :--- | :--- |
| **Vector Space ($M$)** | $L_2(\mathbb{Z}^2)$ | Signals $X$ are sequences (images) with two integer indices $(m, n)$ and finite energy. |
| **Algebra ($A$)** | The algebra of polynomials of two variables, $P(x, y)$ | Filters $h$ are abstract symbolic polynomials involving powers of $x$ and $y$: $h = \sum_{k,l} h_{kl} x^k y^l$. This algebra has two generators, $x$ and $y$. |
| **Homomorphism ($\rho$)** | Defined by mapping the generators $x$ and $y$ to two shift operators, $S_x$ and $S_y$. | $S_x$ shifts the first coordinate: $(S_x X)_{mn} = (X)_{(m-1)n}$, and $S_y$ shifts the second coordinate: $(S_y X)_{mn} = (X)_{m(n-1)}$. The instantiated filter is $\rho(h) = \sum_{k,l} h_{kl} S_x^k S_y^l$. |

---

## Question 24 (2 points)
Consider the algebra of polynomials of one variable. Explain how this algebra is generated by the monomial $t$. Define a shift operator $S$ and explain how the map $\rho(t) = S$ defines a homomorphism. Polynomials on $t$ and polynomials on $S$ represent filters and their instantiations. Explain their similarities and differences.

**Answer:**

**Generation by the Monomial $t$:**
The algebra of polynomials of a single variable, $P(t)$, is generated by the monomial $g = t$. Any filter $h$ belonging to this algebra can be expressed as an element generated by a polynomial on $t$ using the operations (summation, scalar multiplication, and algebra product) defined within the algebra. Because every filter $h = \sum_k h_k t^k$ can be constructed this way, the element $t$ generates the entire algebra.

**Shift Operator and the Homomorphism $\rho(t)=S$:**
*   **Shift Operator ($S$):** Let $(M, \rho)$ be a representation of the algebra $A$. A shift operator $S$ is the image of a generator $g \in A$ under the homomorphism $\rho$, defined as $S = \rho(g)$. For the algebra of polynomials, the generator is $t$, so the shift operator is $S = \rho(t)$. The shift operator $S$ is a concrete linear transformation in the space of endomorphisms $\text{End}(M)$.
*   **Homomorphism $\rho(t)=S$:** A homomorphism $\rho$ is a map that preserves the operations of the algebra (sum, product, scalar product). Since the algebra of polynomials is generated by $t$, defining the image of the generator, $\rho(t) = S$, is sufficient to define the homomorphism for all filters $h \in A$. This relationship ensures that if a filter $h$ is a polynomial $p_A(t)$ on $t$, its instantiation $\rho(h)$ is the equivalent polynomial $p_M(S)$ on $S$.

**Similarities and Differences between Polynomials on $t$ and $S$:**

| Feature | Polynomial on $t$ ($p_A(t)$) | Polynomial on $S$ ($p_M(S)$) |
| :--- | :--- | :--- |
| **Representation** | The abstract definition of the filter. | The concrete instantiation of the filter. |
| **Meaning/Tether** | Untethered to a specific graph or signal model. | Tethered to a specific signal model via the shift operator $S$. |
| **Operations Used** | Operations of the abstract algebra $A$ (summation, algebra product). | Operations of the algebra of Endomorphisms $\text{End}(M)$ (e.g., matrix operations or integral operator composition). |
| **Similarity** | Both use the same filter coefficients $\{h_k\}$. | |

---

## Question 25 (2 points)
The three fundamental concepts of an algebraic signal model are abstract polynomials, polynomials on a shift operator $S$ and polynomials on a scalar variable $\lambda$. Define these three polynomials and explain their equivalence and their different meanings.

**Answer:**
The three polynomials are central components of the ASP model, all sharing the same coefficients, but differing in nature, operations, and meaning.

**Polynomial 1: The Filter (Abstract Polynomial)**
*   **Definition:** $p_A(G) = \sum_{k} h_k g^k$, a polynomial expressed in terms of the generator elements $G$ of the algebra $A$.
*   **Meaning:** This is the abstract definition of a filter. It is untethered to any specific signal model.
*   **Operations:** It uses the operations of the algebra $A$ (sum, product, scalar product).

**Polynomial 2: The Instantiation (Polynomial on a Shift Operator $S$)**
*   **Definition:** $p_M(S) = \sum_{k} h_k S^k$, a polynomial expressed in terms of the shift operators $S$, which are the images $\rho(G)$ in the space of endomorphisms $\text{End}(M)$.
*   **Meaning:** This is the concrete instantiation of the filter. It determines the actual effect the filter has on a signal $x$, and is tethered to a specific signal model $M$.
*   **Operations:** It uses the operations of the algebra of Endomorphisms of $M$ (e.g., matrix operations or integral operator composition).

**Polynomial 3: The Frequency Response (Polynomial on a Scalar Variable $\lambda$)**
*   **Definition:** $p_F(\lambda) = \sum_{k} h_k \lambda^k$, a polynomial function where the variables $\lambda$ take values on the field $F$.
*   **Meaning:** This is a simpler, analytical representation of the filter. It is used as a tool to explain properties like discriminability, stability, and transferability. It is untethered to a specific signal model.
*   **Operations:** It uses only the operations of the field $F$ (product and sum).

**Equivalence:**
The three polynomials are considered equivalent because they are all constructed using the same set of coefficients $\{h_k\}$. They are related through the homomorphism $\rho$ and the spectral representation $\Delta$, even though they exist in different spaces and use different operations.

---

## Question 26 (2 points)
Explain how graph filters and discrete time filters are different instantiations of the same algebraic filters.

**Answer:**
Graph filters (used in Graph Signal Processing, GSP) and discrete time filters (used in Discrete Time Signal Processing, DTSP) are different instantiations of the same abstract algebraic filters because they share the same filter algebra but are realized through different representations.

1.  **Same Algebraic Filter (The Algebra $A$):** Both GSP and DTSP use the algebra of polynomials of a single variable, $t$, denoted $P(t)$. Filters $h$ in this algebra are defined as symbolic polynomials $h = \sum_k h_k t^k$. This polynomial, known as the abstract polynomial or filter ($P_1$), is untethered to any specific signal model.
2.  **Different Instantiations (The Representation $(M, \rho)$):** The algebraic filter $h$ is mapped to a concrete operator $p_M(S)$ using the homomorphism $\rho(t) = S$, where $S$ is the shift operator specific to the vector space $M$.

| Signal Processing Model | Vector Space $M$ (Signals) | Shift Operator $S = \rho(t)$ | Filter Instantiation $p_M(S)$ |
| :--- | :--- | :--- | :--- |
| **Graph Signal Processing (GSP)** | $\mathbb{R}^n$ (Vectors with $n$ components) | Graph Shift Operator (GSO) $S$ (e.g., adjacency matrix) | $H(S) = \sum_k h_k S^k$ |
| **Discrete Time SP (DTSP)** | $L_2(\mathbb{Z})$ (Finite-energy sequences over integers) | Time Shift Operator $S$, where $(SX)_n = (X)_{n-1}$ | $Y_n = \sum_k h_k (X)_{n-k}$ (Time convolution) |

Since the definitions of the filter in the node domain (Polynomial 2, the instantiation) are derived from the same abstract polynomial (Polynomial 1) but operate on different vector spaces and use different shift operators, they are different instantiations of the same abstract algebraic filters.

---

## Question 27 (2 points)
Explain how graphon filters and discrete time filters are different instantiations of the same algebraic filters.

**Answer:**
Graphon filters (used in Graphon Signal Processing, WSP) and discrete time filters (DTSP) are different instantiations of the same algebraic filters for reasons parallel to those explained in Question 26.

1.  **Same Algebraic Filter:** Both WSP and DTSP are based on the algebra of polynomials of a single variable, $t$. The abstract filter is the same polynomial $h = \sum_k h_k t^k$.
2.  **Different Instantiations:** The difference lies in how the generating element $t$ is mapped (homomorphism $\rho$) to a shift operator $S$ in the respective vector spaces $M$.

| Signal Processing Model | Vector Space $M$ (Signals) | Shift Operator $S = \rho(t)$ | Filter Instantiation $p_M(S)$ |
| :--- | :--- | :--- | :--- |
| **Graphon SP (WSP)** | $L_2([0, 1])$ (Finite-energy functions) | Graphon Shift Operator (WSO) $T_W$, an integral operator | $T_H = \sum_k h_k T_W^{(k)}$ |
| **Discrete Time SP (DTSP)** | $L_2(\mathbb{Z})$ (Finite-energy sequences) | Time Shift Operator $S$, where $(SX)_n = (X)_{n-1}$ | $Y_n = \sum_k h_k (X)_{n-k}$ (Time convolution) |

Since the filter coefficients $\{h_k\}$ are identical, the filters possess the same underlying algebraic structure, but their application (instantiation) differs fundamentally based on the domain of the signals ($L_2([0, 1])$ vs. $L_2(\mathbb{Z})$) and the corresponding shift operators ($T_W$ vs. $S$).

---

## Question 28 (2 points)
An algebraic neural network is made up of layers, each of which composes algebraic filters, pointwise nonlinearities and pooling operators. Explain.

**Answer:**
An Algebraic Neural Network (AlgNN) is defined as a stacked layered structure. Each layer $\ell$ relies on a specific algebraic signal model, defined by the triplet $(A_\ell, M_\ell, \rho_\ell)$, and a composition map $\sigma_\ell$.

The operation at layer $\ell$ processes an input $x^{\ell-1}$ to produce an output $x^\ell$. This process composes three primary components:
1.  **Algebraic Filters (Convolution):** The filter operation is represented by $\rho_\ell(a_\ell)$. The filter $a_\ell$ is an element of the algebra $A_\ell$. The homomorphism $\rho_\ell$ instantiates this filter as a concrete linear transformation (convolution) $\rho_\ell(a_\ell)$ applied to the input $x^{\ell-1}$. If the network has multiple features, a family of filters $\rho_\ell(a_\ell^{gf})$ processes the input features $x_{\ell-1}^g$.
2.  **Pointwise Nonlinearities ($\eta_\ell$):** After the convolution/filtering step, the result is processed by the nonlinearity operator $\eta_\ell$. This is assumed to be a pointwise nonlinearity, such as ReLU, sigmoid, or hyperbolic tangent.
3.  **Pooling Operators ($P_\ell$):** The layer structure includes an operator $\sigma_\ell$, which is defined as the composition of the pooling operator $P_\ell$ and the nonlinearity $\eta_\ell$ ($\sigma_\ell = P_\ell \circ \eta_\ell$). The pooling operator $P_\ell$ is responsible for increasing computational efficiency and performance. $P_\ell$ performs a projection, mapping elements from the vector space $M_\ell$ of the current layer to the vector space $M_{\ell+1}$ for the next layer.

The overall output of the AlgNN, $x_L$, is the result of composing these operations over $L$ layers.

---

## Question 29 (2 points)
Given a graphon $W(u, v)$ we define a graphon shift $T_W$. Define a graphon filter to process signals $X(u)$ supported on this graphon. Prove that Graphon filters are pointwise in the Graphon Fourier transform domain.

**Answer:**
A graphon filter is defined by filter coefficients $\{h_k\}$ and the Graphon Shift Operator (WSO), $T_W$. When applied to an input signal $X(u)$, the output signal $Y(v)$ is given by the linear combination of the graphon diffusion sequence:
$$Y(v) = (T_H X)(v) = \sum_{k=0}^K h_k (T_W^{(k)} X)(v)$$

**Proof that Graphon Filters are Pointwise in the Graphon Fourier Transform (WFT) Domain:**
The WSO $T_W$ is a self-adjoint Hilbert-Schmidt operator whose eigenfunctions $\{\phi_j\}$ form a complete orthonormal basis of $L_2([0, 1])$. The action of $T_W$ on a signal $X$ can be expressed in the WFT domain:

1.  **Decomposition of Shift Operator:** The application of the $k$-th power of the WSO, $T_W^{(k)}$, to $X$ is defined by its decomposition using the eigenvalues $\lambda_j$ and eigenfunctions $\phi_j$:
    $$(T_W^{(k)} X)(v) = \sum_{j} \lambda_j^k \phi_j(v) \int_0^1 \phi_j(u)X(u)du$$
    The integral term is the $j$-th component of the WFT of $X$, denoted $\hat{X}_j$.
    $$(T_W^{(k)} X)(v) = \sum_{j} \lambda_j^k \phi_j(v) \hat{X}_j$$
2.  **Output Signal in Spectral Domain:** Substituting this into the definition of the filter $Y(v)$:
    $$Y(v) = \sum_{k=0}^K h_k \left( \sum_{j} \lambda_j^k \phi_j(v) \hat{X}_j \right)$$
    $$Y(v) = \sum_{j} \left( \sum_{k=0}^K h_k \lambda_j^k \right) \phi_j(v) \hat{X}_j$$
3.  **Graphon Fourier Transform of Output:** The $j$-th component of the WFT of the output, $\hat{Y}_j$, is obtained by integrating $Y(v)$ against the eigenfunction $\phi_j(v)$. Due to the orthonormality of the eigenfunctions, the integral $\int_0^1 \phi_i(v) \phi_j(v) dv$ is 1 if $i=j$ and 0 otherwise:
    $$\hat{Y}_j = \int_0^1 Y(v) \phi_j(v) dv = \sum_{i} \left( \sum_{k=0}^K h_k \lambda_i^k \right) \hat{X}_i \int_0^1 \phi_i(v) \phi_j(v) dv$$
    $$\hat{Y}_j = \sum_{k=0}^K h_k \lambda_j^k \hat{X}_j$$
4.  **Pointwise Conclusion:** The term $\sum_{k=0}^K h_k \lambda_j^k$ is the frequency response $h(\lambda_j)$ of the filter. Therefore, the components of the input and output WFTs are related by a simple multiplication:
    $$\hat{Y}_j = h(\lambda_j) \hat{X}_j$$

Since the $j$-th output component depends exclusively on the $j$-th input component and the eigenvalue $\lambda_j$, the graphon filter operates pointwise in the Graphon Fourier Transform domain.

---

## Question 30 (2 points)
We are given a graph $G_n$ and a graph signal $X_n$ supported on this graph. Define induced graphons and induced graphon signals.

**Answer:**
The concepts of induced graphons and induced graphon signals allow a finite-node graph $G_n$ and its associated vector signal $x_n$ to be represented as continuous objects defined on the unit interval $[0, 1]$.
To define both, one first considers a regular partition of the unit interval $[0, 1]$ into $n$ subintervals, $I_i$, where $I_i = [(i-1)/n, i/n)$.

**Induced Graphon ($W_G$):**
An induced graphon $W_G$ is the graphon representation of an undirected graph $G = \{V, E, S\}$ with $n$ nodes and normalized graph shift operator $S$.
The induced graphon $W_G(u, v)$ is defined by mapping the adjacency matrix entries of the graph, $[S]_{ij}$, onto the corresponding sub-squares of the unit square $[0, 1]^2$:
$$W_G(u, v) = [S]_{ij} \times \mathbb{I}(u \in I_i) \mathbb{I}(v \in I_j)$$
where $\mathbb{I}(\cdot)$ is an indicator function. This construction ensures that if $u$ falls into the $i$-th subinterval and $v$ falls into the $j$-th subinterval, the value of the graphon is the weight of the edge between node $i$ and node $j$.

**Induced Graphon Signal ($X_G$):**
An induced graphon signal $X_G$ is the continuous representation of a graph signal $x$ supported on graph $G$.
The induced signal $X_G(u)$ is defined by assigning the scalar value of the signal at node $i$, denoted $x_i$, to the entire $i$-th subinterval $I_i$:
$$X_G(u) = x_i \times \mathbb{I}(u \in I_i)$$
This transforms the discrete vector signal $x$ into a function defined on the unit interval.

Together, the pair $(W_G, X_G)$ constitutes the graphon signal induced by the graph signal $(G, x)$.

---

## Question 31 (2 points)
Given a graphon $W(u, v)$ we define a graphon shift $T_W$ and denote eigenvalues and eigenfunctions of this operator as $\lambda_k$ and $\phi_k$. Eigenvalues of graphons are countable. They are between -1 and 1. And they accumulate at 0. Explain.

**Answer:**
The graphon shift operator (WSO) $T_W$, defined by the graphon $W(u, v)$, is a self-adjoint Hilbert-Schmidt operator. This classification dictates the properties of its spectrum:
1.  **Countable Eigenvalues:** The WSO $T_W$ has an infinite but countable number of eigenvalue-eigenfunction pairs $\{(\lambda_i, \phi_i)\}^\infty_{i=1}$.
2.  **Bounded Range:** Since the graphon $W$ is symmetric and bounded ($0 \leq W(x, y) \leq 1$), the operator $T_W$ is self-adjoint. Consequently, all its eigenvalues are real and lie in the interval $[-1, 1]$.
3.  **Accumulation at 0:** The eigenvalues of a graphon shift operator accumulate at $\lambda = 0$. This means that the eigenvalues $\lambda_j$ converge to zero as the index $j$ tends to $\pm \infty$. This is the only point of accumulation for the eigenvalues. A direct consequence is that for any fixed positive constant $c$, the number of eigenvalues $\lambda_i$ such that $|\lambda_i| \geq c$ is finite ($n_c < \infty$).

---

## Question 32 (2 points)
Given a graphon $W(u, v)$ we define a graphon shift $T_W$ and denote eigenvalues and eigenfunctions of this operator as $\lambda_k$ and $\phi_k$. Use these definitions to define the graphon Fourier transform. Eigenvalues of a graph sequence $\{G_n\}^\infty_{n=1}$ that converges to a graphon $W(u, v)$ in the homomorphism density sense, converge to the graphon eigenvalues. Convergence of eigenvalues could imply convergence of the graph Fourier transform to the graphon Fourier transform. Alas, it does not. Explain why.

**Answer:**

**Definition of the Graphon Fourier Transform (WFT):**
The eigenfunctions $\{\phi_j\}$ of the WSO $T_W$ form a complete orthonormal basis of the signal space $L_2([0, 1])$. The Graphon Fourier Transform (WFT), denoted $\hat{X} = \text{WFT}(X)$, projects a graphon signal $X$ onto this basis.
The components of the WFT, $\hat{X}_j$, are discrete outputs associated with the eigenvalues $\lambda_j$ and are defined by the inner product (integral) of the signal $X$ and the eigenfunction $\phi_j$:
$$\hat{X}_j = \hat{X}(\lambda_j) = \int_0^1 X(u)\phi_j(u)du$$

**Why Convergence of Eigenvalues Does Not Imply Convergence of Transforms:**
A graph sequence $\{G_n\}$ converging to a graphon $W$ implies that the normalized graph eigenvalues converge to the graphon eigenvalues for all indices $j$: $\lim_{n \to \infty} \lambda_j(S_n)/n = \lambda_j(T_W)$.

However, the GFT convergence to the WFT does not hold in general.
1.  **Transform Dependence:** The WFT and the Graph Fourier Transform (GFT) are defined as projections on eigenvectors (or eigenfunctions), not projections merely on eigenvalues. For the transforms to converge, the eigenvectors/eigenfunctions themselves must converge uniformly.
2.  **Non-Uniform Eigenvector Convergence:** Convergence of an eigenvector depends on the eigenvalue marginâ€”how close its associated eigenvalue is to other eigenvalues.
3.  **Accumulation Challenge:** Because graphon eigenvalues accumulate at $\lambda=0$, the margins associated with eigenvalues near zero vanish. This accumulation means that the distance between graph eigenvectors and graphon eigenfunctions is not sufficiently bounded near $\lambda=0$.

**Conclusion:** Since the convergence of the eigenvectors (and thus the GFT and inverse GFT) is not uniform, the convergence of the sequence of GFTs to the WFT cannot be claimed generally, even though the eigenvalues converge.

---

## Question 33 (2 points)
Given a graphon $W(u, v)$, define a graphon shift $T_W$ and denote eigenvalues and eigenfunctions of this operator as $\lambda_k$ and $\phi_k$. Define a graphon filter to process signals $X(u)$ supported on this graphon and define the frequency response of this graphon filter. Explain the role of graphon eigenvalues and eigenvectors in this definition.

**Answer:**

**Graphon Filter Definition:**
The graphon filter $T_H$ of order $K$, specified by coefficients $\{h_k\}$, processes an input signal $X(u)$ by operating on the graphon diffusion sequence (successive applications of the WSO $T_W$):
$$Y(v) = (T_H X)(v) = \sum_{k=0}^K h_k (T_W^{(k)} X)(v)$$

**Frequency Response Definition:**
The filter $T_H$ is pointwise in the WFT domain. The frequency response $h(\lambda)$ of the graphon filter is the polynomial on a scalar variable $\lambda$ determined by the filter coefficients $\{h_k\}$:
$$h(\lambda) = \sum_{k=0}^K h_k \lambda^k$$

**Role of Graphon Eigenvalues and Eigenvectors (Eigenfunctions):**
*   **Role of Eigenvalues ($\lambda_k$):** The eigenvalues $\lambda_k$ represent the frequencies of the graphon signal. In the spectral domain, the output WFT component $\hat{Y}_j$ is obtained by scaling the input component $\hat{X}_j$ by the frequency response evaluated at the eigenvalue $\lambda_j$:
    $$\hat{Y}_j = h(\lambda_j) \hat{X}_j$$
    Thus, the eigenvalues serve as the variable $\lambda$ on which the filter's effect is computed.
*   **Role of Eigenfunctions ($\phi_k$):** The eigenfunctions $\{\phi_k\}$ are the orthonormal basis used to decompose the signal $X$. They facilitate the Graphon Fourier Transform (WFT), which projects the signal from the node domain into the frequency domain. The WFT components $\hat{X}_j$ are derived from the inner product of the signal with the eigenfunctions $\phi_j$.

---

## Question 34 (2 points)
Given a graphon $W(u, v)$, define a graphon shift $T_W$ and denote eigenvalues and eigenfunctions of this operator as $\lambda_k$ and $\phi_k$. Define a graphon filter to process signals $X(u)$ supported on this graphon and define the frequency response of this graphon filter. This is the same definition of the frequency response of a graph filter. Explain why and how we can use this fact to prove convergence of graph filters to graphon filters.

**Answer:**
A graphon shift operator (WSO), denoted $T_W$, is an integral linear operator defined by the graphon $W(u, v)$ applied to a graphon signal $X(u) \in L_2([0, 1])$. The eigenfunctions $\phi_k$ and eigenvalues $\lambda_k$ satisfy the integral equation:
$$(T_W \phi)(v) = \int_0^1 W(u, v) \phi(u) du = \lambda \phi(v)$$

A graphon filter $T_H$ of order $K$, with coefficients $\{h_k\}$, processes an input signal $X(u)$ to produce an output signal $Y(v)$ via a linear combination of the graphon diffusion sequence:
$$Y(v) = (T_H X)(v) = \sum_{k=0}^K h_k (T_W^{(k)} X)(v)$$

The frequency response of the graphon filter, $h(\lambda)$, is defined as the polynomial on a scalar variable $\lambda$ determined by the filter coefficients $\{h_k\}$:
$$h(\lambda) = \sum_{k=0}^K h_k \lambda^k$$

**Explanation of Identical Frequency Response:**
The definition of the frequency response of a graphon filter is exactly the same definition as the frequency response of a graph filter, provided they share the same filter coefficients $\{h_k\}$.
This equivalence exists because both graph filters and graphon filters are derived from the same algebraic filterâ€”the abstract polynomial $p_A(t)$ in the algebra of polynomials of a single variable $t$. This abstract filter defines the filter structure $h(\lambda)$.
*   The graph filter instantiates this polynomial by replacing $t$ with the discrete Graph Shift Operator $S_n$.
*   The graphon filter instantiates this polynomial by replacing $t$ with the continuous Graphon Shift Operator $T_W$.

The frequency response $h(\lambda)$ (Polynomial 3) is a property of the abstract filter (Polynomial 1) and is therefore untethered to the specific instantiation (graph or graphon).

**Using this Fact to Prove Convergence:**
The shared frequency response, combined with the convergence of eigenvalues, is used to prove the convergence of filters in the spectral domain.
1.  **Eigenvalue Convergence:** If a sequence of graphs $\{G_n\}$ converges to a graphon $W$ in the homomorphism density sense, the normalized graph eigenvalues $\lambda_{n,j} = \lambda_j(S_n)/n$ converge to the graphon eigenvalues $\lambda_j(T_W)$ for all indices $j$.
2.  **Continuity Implies Frequency Response Convergence:** Since the frequency response $h(\lambda)$ is a polynomial, it is a continuous function. Because the input (eigenvalues) converges and the function is continuous, the output (frequency representation) must also converge.
3.  **Theorem of Convergence:** The Convergence of graph filter frequency response Theorem formally states that if $\{G_n\} \to W$ and $h$ is continuous, then the frequency representation of the graph filters converges to that of the graphon filter:
    $$\lim_{n \to \infty} \hat{H}_n(\lambda_{n,j}) = \hat{T}_H(\lambda_j)$$

This convergence in the spectral domain is the necessary first step to establish that graph filter outputs converge to graphon filter outputs in the node domain under conditions like Lipschitz continuity or bandlimited inputs.

---

## Question 35 (2 points)
An eigenvector of a graph sequence that converges to a graphon converges to the graphon eigenvector. However, the convergence depends on how close the eigenvalue associated to the eigenvectors is to other eigenvalues. This fact precludes convergence of the graph FT to the graphon FT unless we restrict attention to graphon bandlimited signals. Explain.

**Answer:**

**Dependence of Eigenvector Convergence:**
The statement is true: the convergence of the graph eigenvector $v_{n,j}$ (or induced eigenfunction $\phi_{n,j}$) to the graphon eigenfunction $\phi_j$ depends on the eigenvalue margin, $d(\lambda_j, \lambda_{n,j})$. This margin measures the distance between the eigenvalue pair $(\lambda_j, \lambda_{n,j})$ and all other eigenvalues of the respective operators.
The distance between eigenfunctions is bounded inversely by this margin (Davis-Kahan theorem):
$$\|\phi_j - \phi_{n,j}\| \leq \frac{\pi}{2} \frac{\|W - W_{G_n}\|}{d(\lambda_j, \lambda_{n,j})}$$

**Why Convergence Fails Generally:**
The convergence of the Graph Fourier Transform (GFT) to the Graphon Fourier Transform (WFT) fails generally because the GFT/WFT are projections onto eigenvectors/eigenfunctions, not just eigenvalues.
1.  **Eigenvalue Accumulation:** Graphon eigenvalues accumulate at $\lambda = 0$. There is an infinite, countable number of eigenvalues clustered around zero.
2.  **Vanishing Margin:** Because of this clustering, the eigenvalue margin $d(\lambda_j, \lambda_{n,j})$ vanishes for eigenvalues close to zero.
3.  **Non-uniform Convergence:** As the margin approaches zero, the bound on the distance between the eigenfunctions explodes. This means that the convergence of eigenvectors near $\lambda=0$ is not uniform, precluding the general convergence of the GFT to the WFT.

**Resolution via Graphon Bandlimited Signals:**
Convergence is achieved if we restrict attention to graphon bandlimited signals.
*   A graphon signal $(W, X)$ is defined as $c$-bandlimited if its WFT coefficients $\hat{X}(\lambda_j)$ are zero for all eigenvalues $\lambda_j$ such that $|\lambda_j| < c$, where $c > 0$ is the bandwidth limit.
*   By restricting the signal to be $c$-bandlimited, we effectively eliminate the components associated with the region around $\lambda=0$ where eigenvalues accumulate and the margin vanishes.
*   With the problem components removed, the remaining eigenvalues are separated by a sufficient margin, allowing the GFT of the graph signal sequence to converge to the WFT of the graphon signal.

---

## Question 36 (2 points)
We are given a time series $X$ with entries $x_t$ with $t = 0, \ldots, T - 1$. Define linear attention coefficients $B_{tu}$ as well as the linear attention matrix $B$.

**Answer:**
The linear attention mechanism relies on defining queries ($Q x_t$) and keys ($K x_u$), where $Q$ is the query matrix and $K$ is the key matrix.

**Linear Attention Coefficients ($B_{tu}$):**
The linear attention coefficient $B_{tu}$ is used to measure the similarity between two components $x_t$ and $x_u$ of the time series. It is defined as the inner product between the query vector for $x_t$ and the key vector for $x_u$:
$$B_{tu} = \langle Q x_t, K x_u \rangle = (Q x_t)^T (K x_u)$$

**Linear Attention Matrix ($B$):**
The attention coefficients $B_{tu}$ can be grouped into a single attention matrix $B$. This matrix is given by the outer product between the query matrix $QX$ (where $X$ is the time series matrix) and the key matrix $KX$:
$$B = (QX)^T (KX)$$
This matrix $B$ contains a large number of attention coefficients, but they are generated by a relatively small number of parameters contained in $Q$ and $K$.

---

## Question 37 (2 points)
We are given a time series $X$ with entries $x_t$ with $t = 0, \ldots, T - 1$. Define the softmax nonlinear attention matrix $A$.

**Answer:**
The softmax attention mechanism post-processes the linear attention coefficients $B_{tu}$ with a non-linear function.

1.  **Linear Attention Matrix:** First, the linear attention matrix $B$ is calculated based on the query matrix $Q$ and key matrix $K$:
    $$B = (QX)^T (KX)$$
2.  **Softmax Application:** The softmax nonlinear attention matrix $A$ is defined by applying the softmax function, denoted $\text{sm}(\cdot)$, to the linear attention matrix $B$. The softmax function implements normalization along the rows of $B$:
    $$A = \text{sm} \left( (QX)^T (KX) \right)$$

If $A_{tu}$ denotes the entries of $A$, the softmax operation ensures that the entries of each row sum to 1. The coefficient $A_{tu}$ is obtained by applying an exponential pointwise nonlinearity to the linear coefficient $B_{tu}$, followed by normalization across all indices $u'$:
$$A_{tu} = \frac{\exp(B_{tu})}{\sum_{u'=0}^{T} \exp(B_{tu'})}$$

---

## Question 38 (2 points)
We are given a time series $X$ with entries $x_t$ with $t = 0, \ldots, T - 1$ along with an attention matrix $A$. Use $A$ to define contextual representations $y_t$.

**Answer:**
The attention coefficients $A_{tu}$ are used to create a contextual representation $z_t$ (before recovery) or $y_t$ (after recovery) of $x_t$. This representation is a linear combination of all vectors in the time series, weighted by their relevance (the attention coefficient $A_{tu}$).

1.  **Contextual Representation ($z_t$):** This is defined as a weighted sum of the time series vectors $x_u$ multiplied by a value matrix $V$:
    $$z_t = \sum_{u=0}^T (V x_u) A_{tu}$$
    The vectors $V x_u$ represent the time series in a lower-dimensional space.
2.  **Dimensionality Recovery ($y_t$):** The final output of the attention layer, $y_t$, recovers the original dimensionality ($n$) by multiplying the reduced-dimension contextual representation $z_t$ by the transpose of a recovery matrix $W$:
    $$y_t = W^T z_t$$

Substituting the definition of $z_t$ into the recovery step yields the definition of $y_t$ using the attention matrix $A$:
$$y_t = W^T \sum_{u=0}^T A_{tu} V x_u$$
The vector $y_t$ is a contextual representation that has the same dimensionality as the components of the time series.

---

## Question 39 (2 points)
We are given a time series $X$ with entries $x_t$ with $t = 0, \ldots, T - 1$. Define a softmax attention layer with output $Y$. Explain the postprocessing of the entries $y_t$ of $Y$ with a local nonlinear operation.

**Answer:**
The output $Y$ of a softmax attention layer is the matrix containing all contextual representations $y_t$. The output $Y$ is defined in matrix form as:
$$Y = W^T V X A^T$$

**Postprocessing of Entries $y_t$:**
It is common practice to postprocess the contextual representation $Y$ further. This postprocessing is performed without further mixing different time components; it acts locally on each $y_t$.
Two common local nonlinear postprocessing methods are used for the entries $y_t$ (which form the columns of $Y$):
1.  **Pointwise Nonlinearity:** The simplest approach is to apply a pointwise nonlinearity, denoted $\eta$ or $\sigma$, to the entire output matrix $Y$:
    $$Y^{\prime} = \eta(Y)$$
    This means the nonlinearity is applied element-wise to every entry of every vector $y_t$.
2.  **Fully Connected Neural Network (FCNN):** Alternatively, each individual output vector $y_t$ can be postprocessed with a fully connected neural network (FCNN). This is acceptable because the vectors $y_t$ are typically not large, and the same FCNN is used across all time steps $t$.

---

## Question 40 (2 points)
Define a transformer with $L$ layers in which contextual representations at each layer are postprocessed with a local fully connected neural network.

**Answer:**
A Transformer is a layered architecture defined by a recursion composed of attention and postprocessing.
A standard attention layer at layer $\ell$ computes the softmax attention matrix $A_\ell$ and the contextual representation matrix $Y_\ell$, which contains the representations $y_t$.

The recursion is initialized with $X_0 = X$ and repeated $L$ times. For a single head, the process is:
1.  **Softmax Attention Matrix Calculation ($A_\ell$):**
    $$A_\ell = \text{sm} \left( (Q_\ell X_{\ell-1})^T (K_\ell X_{\ell-1}) \right)$$
2.  **Contextual Representation Calculation ($Y_\ell$):**
    $$Y_\ell = W_\ell^T V_\ell X_{\ell-1} A_\ell^T$$
3.  **Postprocessing with Local FCNN ($X_\ell$):** Instead of the common postprocessing step $X_\ell = \sigma(Y_\ell)$ using a pointwise nonlinearity, the output $X_\ell$ is generated by postprocessing the entries $y_t$ of $Y_\ell$ with a local fully connected neural network (FCNN). This operation is performed without further mixing different time components.

The overall output of the Transformer is $Y_L$, the output signal at the last layer $L$.

---

## Question 41 (2 points)
Define a transformer with $L$ layers and $H$ heads per layer in which contextual representations at each layer are postprocessed with a local pointwise nonlinearity.

**Answer:**
A transformer with multiple features (heads) per layer uses multihead attention. The recursion is initialized with $X_0 = X$ and repeated $L$ times. For a layer $\ell$:

1.  **Multihead Attention Calculation ($A_\ell^h, Y_\ell^h$):** For each head $h = 1, \ldots, H$, the softmax attention matrix $A_\ell^h$ and the contextual representation $Y_\ell^h$ are computed using separate query ($Q_\ell^h$), key ($K_\ell^h$), value ($V_\ell^h$), and recovery ($W_\ell^h$) matrices:
    $$A_\ell^h = \text{sm} \left( (Q_\ell^h X_{\ell-1})^T (K_\ell^h X_{\ell-1}) \right)$$
    $$Y_\ell^h = (W_\ell^h)^T V_\ell^h X_{\ell-1} (A_\ell^h)^T$$
2.  **Combination and Postprocessing ($X_\ell$):** Unlike graph neural networks (GNNs), the output of attention layers always has a single feature. The output of the layer $X_\ell$ is obtained by summing the multiple features (outputs $Y_\ell^h$) generated by the different heads and then applying a local pointwise nonlinearity $\sigma$ (or $\eta$):
    $$X_\ell = \sigma \left( \sum_{h=1}^H Y_\ell^h \right)$$

---

## Question 42 (2 points)
State a theorem claiming the stability of algebraic filters to deformations of the shift operators. The introduction of this theorem requires that you introduce some preliminary definitions. Do not dwell on the definitions of Frechet derivatives and commutativity factors. Do dwell on the definition of Lipschitz and integral Lipschitz filters to explain the significance of this theorem. The implications of this theorem are the same implications that we earlier observed for graph filters. What is the purpose then of introducing this new theorem?

**Answer:**
The stability of algebraic filters is guaranteed by imposing constraints on the filter's frequency response.

**Preliminary Definitions:**
*   **Lipschitz Filter ($L_0$):** A polynomial $p(\lambda)$ is said to be Lipschitz if there exists a constant $L_0 > 0$ such that for all $\lambda, \mu \in \mathbb{C}$, the magnitude of the difference between $p(\lambda)$ and $p(\mu)$ is bounded by the difference between $\lambda$ and $\mu$ scaled by $L_0$.
    $$|p(\lambda) - p(\mu)| \leq L_0 |\lambda - \mu|$$
*   **Integral Lipschitz Filter ($L_1$):** A polynomial $p(\lambda)$ is said to be Integral Lipschitz if there exists a constant $L_1 > 0$ such that the norm of the product of $\lambda$ and the derivative of $p(\lambda)$ is bounded by $L_1$.
    $$\left|\lambda \frac{dp(\lambda)}{d\lambda}\right| \leq L_1$$

**Theorem Statement (Stability of Algebraic Filters):**
Let $A$ be an algebra generated by a single element $g$. Let $\rho(g) = S$ and $\rho(\tilde{g}) = \tilde{S}$ be the shift operator and its perturbed version, respectively. The theorem states that if the filter's abstract polynomial $p_A$ belongs to the set of Lipschitz filters ($A_{L_0}$) and Integral Lipschitz filters ($A_{L_1}$), then the operator $p(S)$ is stable:
$$|p(S)x - p(\tilde{S})x| \leq [(1 + \delta) (L_0 \sup_S |T(S)| + L_1 \sup_S |DT(S)|) + O(|T(S)|^2)] |x|$$

**Significance of Lipschitz Conditions:**
The theorem establishes that algebraic filters can be made stable to perturbations provided these filter constraints are met. The bounds $L_0$ and $L_1$ measure the filter's variability.
The Integral Lipschitz constant $L_1$ ensures that the filter's frequency response becomes flat for large $\lambda$, controlling the length of the interval in which the filter can vary. This condition is crucial for filter stability.
The stability-discriminability tradeoff remains apparent: the stability bound is directly proportional to $L_0$ and $L_1$. A more discriminative filter requires larger variability (larger $L_0$ and $L_1$), which results in a looser, less tight stability bound.

**Purpose of the New Theorem:**
The algebraic stability theorem is important despite having similar implications to prior graph filter analyses (e.g., the transferability bounds) because Algebraic Signal Processing (ASP) provides a generic abstraction of convolutional processing.
1.  **General Framework:** The theorem establishes a unified stability framework that holds for all particular cases of convolutional filters, including Graph Signal Processing (GSP), Graphon Signal Processing (WSP), Discrete Time Signal Processing (DTSP), and Image Processing (IP).
2.  **Inherent Structure:** It demonstrates that the observed stability vs. discriminability tradeoff is inherent to the convolutional algebraic structure itself, rather than being unique to the graph domain.

---

## Question 43 (2 points)
State a theorem claiming the stability of algebraic filters to deformations of the shift operators. The introduction of this theorem requires that you introduce some preliminary definitions. Do not dwell on the definitions of Frechet derivatives and commutativity factors. Do dwell on the definition of Lipschitz and integral Lipschitz filters to explain the significance of this theorem. What does this theorem imply about the stability of discrete time convolutional filters?

**Answer:**
The preliminary definitions and the statement of the **Stability of Algebraic Filters Theorem** are provided in the response to Question 42.

**Implications for Discrete Time Convolutional Filters (DTSP):**
Discrete Time Signal Processing (DTSP) is a particular case of ASP. In DTSP, signals are sequences $X \in L_2(\mathbb{Z})$ processed using filters that are polynomials on the time shift operator $S$. The filter instantiation is convolution: $(Y)_n = \sum_k h_k (X)_{n-k}$.

The algebraic stability theorem implies that DTSP convolutional filters are stable to perturbations of the time shift operator provided the filter's frequency response $h(\lambda)$ is Lipschitz ($L_0$) and Integral Lipschitz ($L_1$).

**Stability Meaning:**
If the underlying algebra (which imposes translation equivariance) is perturbed such that the shift operator $S$ becomes $\tilde{S} = S + T(S)$ (representing quasi-translation equivariance), the output difference $|p(S)x - p(\tilde{S})x|$ will be bounded proportionally to the filter's variability ($L_0, L_1$) and the magnitude of the perturbation ($|T(S)|, |DT(S)|$). This confirms that stability in DTSP is guaranteed by bounded filter variability.

---

## Question 44 (2 points)
State a theorem claiming the stability of algebraic filters to deformations of the shift operators. The introduction of this theorem requires that you introduce some preliminary definitions. Do not dwell on the definitions of Frechet derivatives and commutativity factors. Do dwell on the definition of Lipschitz and integral Lipschitz filters to explain the significance of this theorem. What does this theorem imply about the stability of graphon filters?

**Answer:**
The preliminary definitions and the statement of the **Stability of Algebraic Filters Theorem** are provided in the response to Question 42.

**Implications for Graphon Filters (WSP):**
Graphon Signal Processing (WSP) is a particular case of ASP. Graphon signals $X \in L_2([0, 1])$ are processed by filters that are polynomials on the Graphon Shift Operator (WSO) $T_W$.

The algebraic stability theorem implies that graphon filters $T_H$ are stable to deformations of the graphon shift operator provided the filter's frequency response $h(\lambda)$ is Lipschitz ($L_0$) and Integral Lipschitz ($L_1$).

**Stability Meaning:**
Stability means that if the graphon $W$ (which defines $T_W$) is perturbed to $W'$, resulting in a perturbed shift operator $\tilde{T}_W = T_W + T(T_W)$, the difference between the outputs $|p(T_W)X - p(\tilde{T}_W)X|$ remains bounded by the filter's Lipschitz constants ($L_0, L_1$) and the magnitude of the operator perturbation. This stability is crucial for analyzing the transferability and limit behavior of graph filters sampled from the graphon.

---

## Question 45 (2 points)
State a theorem claiming the stability of algebraic neural networks to deformations of the shift operators. The introduction of this theorem requires that you introduce some preliminary definitions. Do not dwell on the definitions of Frechet derivatives and commutativity factors. Do dwell on the definition of Lipschitz and integral Lipschitz filters to explain the significance of this theorem. The implications of this theorem are the same implications that we earlier observed for graph neural networks. What is the purpose then of introducing this new theorem?

**Answer:**
The stability of Algebraic Neural Networks (AlgNNs) to deformations of the shift operators relies on characterizing the properties of the convolutional filters used within the network layers.

**Preliminary Definitions:**
*   **Lipschitz Filter ($L_0$):** A polynomial $p(\lambda)$ is defined as Lipschitz if there exists a constant $L_0 > 0$ such that for all $\lambda, \mu$, the magnitude of the difference between the polynomial evaluated at those points is bounded by $L_0$ times the distance between the points:
    $$|p(\lambda) - p(\mu)| \leq L_0 |\lambda - \mu|$$
*   **Integral Lipschitz Filter ($L_1$):** A polynomial $p(\lambda)$ is defined as Integral Lipschitz if there exists a constant $L_1 > 0$ such that the magnitude of the product of $\lambda$ and the derivative of $p(\lambda)$ is bounded by $L_1$:
    $$\left|\lambda \frac{dp(\lambda)}{d\lambda}\right| \leq L_1$$

**Theorem Statement (Stability of Algebraic Neural Networks, Multi-Layer):**
Let $\Phi(S, x)$ and $\Phi(\tilde{S}, x)$ be the operators associated with an Algebraic Neural Network (AlgNN) on $L$ layers. If the layer filters are Lipschitz and Integral Lipschitz, the stability is bounded by:
$$\left\|\Phi(S, x) - \Phi(\tilde{S}, x)\right\| \leq L \left[ (1 + \delta) \left( L_0 \sup_{S} |T(S)| + L_1 \sup_{S} |DT(S)| \right) + O(|T(S)|^2) \right] \|x\|$$
where $\tilde{S} = S + T(S)$ is the perturbed shift operator, $L_0$ and $L_1$ are the Lipschitz and Integral Lipschitz constants, and $\delta$ is the commutativity factor.

**Significance of Lipschitz and Integral Lipschitz Conditions:**
The theorem implies that AlgNNs can be made stable to perturbations of the shift operator $S$ if their convolutional filters satisfy these conditions.
The constants $L_0$ and $L_1$ measure the filter's variability.
The Integral Lipschitz condition ($L_1$) ensures that the filter's rate of variation approaches zero for large eigenvalues ($\lambda$), effectively controlling the length of the interval in which the filter can vary. This is essential for bounding the stability error.
The resulting bound shows a stability-discriminability tradeoff: the stability constant is directly proportional to $L_0$ and $L_1$. A filter designed to be highly discriminative (requiring large $L_0$ and $L_1$) will necessarily have a looser stability bound.

**Purpose of Introducing this New Theorem:**
The implications of this theorem regarding the stability vs. discriminability tradeoff are the same as those observed earlier for Graph Neural Networks (GNNs).
The purpose of introducing this theorem in the generic framework of Algebraic Neural Networks (AlgNNs) is to provide a general abstraction that proves that the stability properties are inherent to the convolutional algebraic structure itself, regardless of the specific signal domain (graph, graphon, discrete time, etc.).

---

## Question 46 (2 points)
State a theorem claiming the stability of algebraic neural networks to deformations of the shift operators. The introduction of this theorem requires that you introduce some preliminary definitions. Do not dwell on the definitions of Frechet derivatives and commutativity factors. Do dwell on the definition of Lipschitz and integral Lipschitz filters to explain the significance of this theorem. What does this theorem imply about the stability of discrete time (standard) convolutional neural networks?

**Answer:**
The preliminary definitions (Lipschitz and Integral Lipschitz filters) and the statement of the **Stability of Algebraic Neural Networks, Multi-Layer Theorem** are provided in the response to Question 45.

**Implications for Discrete Time (Standard) Convolutional Neural Networks (CNNs):**
Discrete Time Signal Processing (DTSP) is a particular case of Algebraic Signal Processing (ASP). Standard CNNs (which operate on sequences, like DTSP) are Algebraic Neural Networks (AlgNNs) where the algebraic model is typically defined by the algebra of polynomials $A = P(t)$ modulo $t^N-1$ and the shift operator $S$ is the cyclic shift operator $C$.

The AlgNN stability theorem implies that discrete time convolutional neural networks (CNNs) are stable to perturbations of the time shift operator (quasi-translation equivariance).
1.  If the filters (convolution kernels) used in the layers of the CNN have frequency responses that are Lipschitz ($L_0$) and Integral Lipschitz ($L_1$), the CNN's output stability to deformations in the time shift operator is guaranteed.
2.  The bound on the instability is linear in the number of layers $L$.
3.  This means that small deformations in the fundamental translation-equivariance structure of the time series data result in a bounded change in the CNN's output, provided the filter variability (measured by $L_0$ and $L_1$) is controlled.

---

## Question 47 (2 points)
State a theorem claiming the stability of algebraic neural networks to deformations of the shift operators. The introduction of this theorem requires that you introduce some preliminary definitions. Do not dwell on the definitions of Frechet derivatives and commutativity factors. Do dwell on the definition of Lipschitz and integral Lipschitz filters to explain the significance of this theorem. What does this theorem imply about the stability of graphon neural networks?

**Answer:**
The preliminary definitions (Lipschitz and Integral Lipschitz filters) and the statement of the **Stability of Algebraic Neural Networks, Multi-Layer Theorem** are provided in the response to Question 45.

**Implications for Graphon Neural Networks (WNNs):**
Graphon Signal Processing (WSP) is a particular case of ASP where the vector space is $L_2([0, 1])$ and the shift operator is the Graphon Shift Operator $T_W$. A Graphon Neural Network (WNN) is an AlgNN operating on this continuous domain.

The AlgNN stability theorem implies that Graphon Neural Networks (WNNs) are stable to deformations of the Graphon Shift Operator $T_W$ (i.e., perturbations of the underlying graphon $W$).
1.  If the graphon filters (convolution kernels) used in the WNN layers have frequency responses that are Lipschitz ($L_0$) and Integral Lipschitz ($L_1$), the stability of the WNN output to small changes in the graphon structure is guaranteed.
2.  The stability bound is linear in the number of layers $L$.
3.  The stability of WNNs is particularly important because WNNs are defined as limit objects and generative models for Graph Neural Networks (GNNs). The WNN stability theorem confirms that the stability properties observed in finite GNNs (which are known to inherit stability from graph filters) are preserved in the continuous limit, strengthening the theoretical foundation for GNN transferability and approximation.

---

## Question 48 (2 points)
We presented in this class stability theorems for algebraic filters and algebraic neural networks. Both of these theorems claim the same stability bound. However, the stability vs discriminability tradeoff of algebraic neural networks and algebraic filters may be different. Explain how. Explain the implications of this discussion for discrete time (standard) convolutional filters and neural networks.

**Answer:**
**Stability Bound and the Role of Nonlinearity:**
The stability theorems for both algebraic filters (AFs) and algebraic neural networks (AlgNNs) claim the same core stability bound for a single layer, which is scaled linearly by the number of layers $L$ in the multi-layer case. The stability bound is proportional to the filter's variability, as measured by its Lipschitz constant ($L_0$) and Integral Lipschitz constant ($L_1$).

**Difference in the Stability vs. Discriminability Tradeoff:**
The stability vs. discriminability tradeoff is realized differently between AFs and AlgNNs because AlgNNs include pointwise nonlinearities.
1.  **Algebraic Filters (AFs):** A single algebraic filter must achieve both stability and discriminability in one operation. Stability requires low variability (small $L_0$ and $L_1$) to ensure tight stability bounds. However, high discriminability (sharpness) requires high variability (large $L_0$ and $L_1$). This results in a fundamental non-tradeoff where increased discriminability immediately leads to an exponentially looser, potentially useless, stability bound.
2.  **Algebraic Neural Networks (AlgNNs):** The multi-layer AlgNN structure, coupled with pointwise nonlinearities ($\sigma$), helps to alleviate this filter-level constraint. While the filters at individual layers still face the stability constraint and might "lose discriminability", the nonlinearity mixes frequency components. This mixing allows the network to recover discriminability in subsequent layers. Thus, AlgNNs can manage a tradeoff, whereas simple AFs face a non-tradeoff.

**Implications for Discrete Time (Standard) Convolutional Filters and Neural Networks:**
Discrete Time Signal Processing (DTSP) uses standard convolutional filters (algebraic filters) and Convolutional Neural Networks (CNNs) (algebraic neural networks) as particular cases of the general ASP framework.
1.  **Discrete Time Convolutional Filters (AFs):** Standard convolutional filters are subject to the inherent stability-discriminability non-tradeoff.
    *   For a convolution filter to be highly discriminative (e.g., a sharp band-pass filter), its frequency response must be highly variable, leading to large Lipschitz constants ($L_0, L_1$).
    *   This large variability causes the stability bound to increase, meaning the filter is unstable to small perturbations in the time shift operator $S$ (quasi-translation equivariance).
2.  **Standard Convolutional Neural Networks (CNNs) (AlgNNs):** CNNs, through their multi-layered structure with nonlinearities, can mitigate this inherent constraint.
    *   The nonlinearity in a CNN allows the network to maintain stable individual layers while achieving overall high discriminability across the network depth.
    *   By combining multiple convolution layers and nonlinearities, the network achieves complex discriminative functions while adhering to stability requirements that are proportional to the number of layers $L$.

---

## Question 49 (2 points)
We presented in this class stability theorems for algebraic filters and algebraic neural networks. Both of these theorems claim the same stability bound. However, the stability vs discriminability tradeoff of algebraic neural networks and algebraic filters may be different. Explain how. Explain the implications of this discussion for graphon filters and neural networks.

**Answer:**
**Stability Bound and the Role of Nonlinearity:**
As discussed in Question 48, the core stability bound for algebraic filters and algebraic neural network layers is mathematically the same. The difference in the stability vs. discriminability tradeoff arises because Algebraic Neural Networks use pointwise nonlinearities to mix frequency components, allowing the network to achieve discriminability without sacrificing stability entirely.

**Implications for Graphon Filters and Neural Networks:**
Graphon filters (AFs) and Graphon Neural Networks (WNNs) are particular cases of Algebraic Filters and AlgNNs, respectively, used to analyze the transferability of graph processing models.
1.  **Graphon Filters (AFs):** Graphon filters encounter the stability-discriminability non-tradeoff in the context of transferability, particularly because graphon eigenvalues accumulate at $\lambda=0$.
    *   To ensure high transferability (convergence/stability), the filter must be Lipschitz continuous (low $L_2$) to bound the variability near $\lambda=0$ where the eigenvalues cluster.
    *   This requirement imposes a limitation: the filter cannot discriminate between spectral components associated with eigenvalues close to zero.
    *   This confirms the transferability vs. discriminability non-tradeoff: stability/transferability requires wide, non-discriminative filters, while discriminability requires sharp, non-transferable filters. Filters that transfer well are less discriminative.
2.  **Graphon Neural Networks (WNNs):** WNNs (and their finite-graph counterparts, GNNs) are designed to alleviate this non-tradeoff by using nonlinearities.
    *   The GNN/WNN transferability bound is an extrapolation of the filter approximation bound, scaled by the network depth $L$ and width $F$.
    *   At each layer of the GNN/WNN, the nonlinearity ($\sigma$) scatters eigenvalues across the spectrum. This scattering effect, particularly involving components associated with eigenvalues $|\lambda| \leq c$, allows the WNN to effectively decrease the required bandwidth limit $c$ and increase the filter variability $L_2$ without exploding the transferability bound.
    *   The result is that for the same level of discriminability, GNNs are more transferable than simple graph filters. The nonlinearity transforms the strict non-tradeoff into a manageable tradeoff.
