# Graph Neural Networks

## Table of Contents

- [Course Information](#course-information)
- [Course Objectives](#course-objectives)
- [Introduction](#introduction)
- [Lecture 1: Foundations](#lecture-1-foundations)
  - [Part I: Why Graphs Matter](#part-i-why-graphs-matter)
  - [Part II: Implementation Challenges](#part-ii-implementation-challenges)
  - [Part III: Graph Convolutions](#part-iii-graph-convolutions)
  - [Part IV: From CNNs to GNNs](#part-iv-from-cnns-to-gnns)
- [Course Structure](#course-structure)
- [References](#references)

## Course Information

- **Instructor:** Prof. Alejandro Ribeiro
- **Department:** Electrical and Systems Engineering
- **Institution:** University of Pennsylvania
- **Contact:** aribeiro@seas.upenn.edu
- **Lab Website:** [alelab.seas.upenn.edu](http://alelab.seas.upenn.edu)

## Course Objectives

### Primary Goals

1. **Practical Application** - Develop the ability to use Graph Neural Networks in real-world applications
2. **Theoretical Understanding** - Master the fundamental properties and principles of GNNs

### Learning Outcomes

Upon completion, students will be able to:
- Identify situations where GNNs have potential
- Formulate machine learning problems on graphs using GNNs
- Train and evaluate GNN models effectively
- Understand when and why GNNs work (or don't work)

## Introduction

Graph Neural Networks (GNNs) are the tool of choice for machine learning on graphs. This course explores both the practical applications and theoretical foundations of GNNs, with emphasis on understanding why graphs are pervasive in information processing and how GNNs leverage graph structure for scalable machine learning.

### Key Applications

| Application | Description | Reference |
|------------|-------------|-----------|
| **Authorship Attribution** | Identify authors of anonymous texts using word adjacency networks | Segarra et al. 2014 |
| **Recommendation Systems** | Predict product ratings using customer similarity graphs | Ruiz et al. 2018 |
| **Wireless Networks** | Optimize resource allocation in communication networks | Eisen-Ribeiro 2019 |
| **Autonomous Systems** | Decentralized control of robot swarms | Tolstaya et al. 2019 |

## Lecture 1: Foundations

### Overview

This lecture covers four fundamental topics:
1. **Why** - The importance of machine learning on graphs
2. **How** - Implementation approaches and challenges
3. **Convolutions** - Generalizing from Euclidean space to graphs
4. **Neural Networks** - Evolution from CNNs to GNNs

---

## Part I: Why Graphs Matter

### Graphs as Generic Models of Signal Structure

Graphs provide a universal framework for modeling relationships and structure in data, enabling machine learning in diverse domains.

### Case Study: Authorship Attribution

**Word Adjacency Networks (WANs)**
- **Nodes:** Function words (prepositions, conjunctions)
- **Edges:** Co-occurrence frequency between word pairs
- **Purpose:** Capture grammatical patterns unique to authors

This approach successfully distinguished between Shakespeare and Marlowe's writing styles, even identifying their collaboration in Henry VI.

### Case Study: Recommendation Systems

**Collaborative Filtering**
- **Nodes:** Customers
- **Edges:** Similarity scores based on rating histories
- **Goal:** Predict unrated products by reducing rating variability

The graph structure enables prediction by leveraging similarity patterns among users.

### Multiagent Physical Systems

In physical systems with multiple agents, graphs are not just data structures but inherent system components:

**Decentralized Control**
- Coordinate drone teams without central authority
- Physical proximity determines information availability
- Graph models agent interactions and communication constraints

**Wireless Networks**
- Manage interference in resource allocation
- Radio propagation creates interference patterns
- Graph captures channel relationships between transceivers

> **Key Insight:** In multiagent systems, graphs create tension between local information and global objectives

---

## Part II: Implementation Challenges

### The Scaling Problem

**Challenge:** Generic (fully connected) neural networks don't scale with input dimensionality

**Solution Path:**
- CNNs successfully scale for images through convolutions
- Need to generalize convolution concept to arbitrary graphs

### Architectural Components

CNNs consist of three main components:

| Component | Generalizable to Graphs? | Notes |
|-----------|-------------------------|--------|
| Layers | ✅ Yes | Straightforward generalization |
| Pointwise Nonlinearities | ✅ Yes | Directly applicable |
| Convolutional Filters | ❓ Needs work | Requires mathematical generalization |

### Solution Roadmap

1. Generalize convolutions to graphs → Graph filters
2. Combine with pointwise nonlinearities → Graph filter banks
3. Stack in layers → Graph Neural Networks

---

## Part III: Graph Convolutions

### Key Insight: Convolutions ARE Graph Operations

Traditional convolutions can be expressed as operations on specific graphs:
- **Time signals** → Line graphs
- **Images** → Grid graphs

### Mathematical Framework

#### Time Convolution as Matrix Polynomial

For a signal **x** on a line graph with adjacency matrix **S**:

```
z = h₀x + h₁Sx + h₂S²x + h₃S³x + ... = Σ(k=0 to K-1) hₖSᵏx
```

Where:
- `hₖ` = filter coefficients
- `Sᵏx` = k-times shifted signal

#### Spatial Convolution

For images on a grid graph:

```
z = h₀x + h₁Sx + h₂S²x + h₃S³x + ... = Σ(k=0 to K-1) hₖSᵏx
```

Where `Sᵏx` represents k-times diffused signal.

### Generalization to Arbitrary Graphs

For any graph with adjacency matrix **S** and signal **x**:

```
z = Σ(k=0 to K-1) hₖSᵏx
```

This definition:
- Preserves locality of operations
- Recovers traditional convolutions as special cases
- Enables processing of arbitrary graph-structured data

### Graph Signal Examples

| Domain | Nodes | Signal Values | Edges |
|--------|-------|---------------|-------|
| Recommender Systems | Customers | Product ratings | User similarities |
| Autonomous Systems | Drones | Velocities | Communication range |
| Wireless Networks | Transceivers | QoS requirements | Channel strength |

---

## Part IV: From CNNs to GNNs

### Neural Network Evolution

#### Fully Connected Neural Networks

```
Layer l: xˡ = σ(Hˡxˡ⁻¹)
```
- **H**: Linear transformation matrix
- **σ**: Pointwise nonlinearity
- **Problem**: Doesn't scale

#### Convolutional Neural Networks

```
Layer l: xˡ = σ(hˡ * xˡ⁻¹)
```
- **h**: Convolutional filter
- **Success**: Scales well for images/time series

#### Graph Neural Networks

```
Layer l: xˡ = σ(Σ(k=0 to K-1) hₖˡSᵏxˡ⁻¹)
```
- **S**: Graph adjacency matrix
- **hₖˡ**: Layer l filter coefficients
- **Result**: Scalable processing for arbitrary graphs

### GNN Architecture Properties

- **Generalization of CNNs** - Recovers CNNs when S is a line/grid graph
- **Local operations** - Leverages graph structure for efficiency
- **Scalability** - Growing empirical evidence of success
- **Theoretical foundation** - Minor variation of graph filters

---

## Course Structure

### Theoretical Topics

#### Architectures
- Graph filters and filter banks
- Graph neural networks (detailed)
- Graph recurrent neural networks

#### Fundamental Properties
1. **Permutation Equivariance** - Consistency under node reordering
2. **Stability to Deformations** - Robustness to graph perturbations
3. **Transferability** - Generalization across different graphs

### Laboratory Assignments

| Lab | Topic | Focus |
|-----|-------|-------|
| **Lab 1** | Statistical & Empirical Risk Minimization | Warmup, learning parameterizations |
| **Lab 2** | Recommendation Systems | Compare FC networks, graph filters, GNNs |
| **Lab 3** | Distributed Control | Multiagent systems, decentralized policies |
| **Lab 4** | Wireless Resource Allocation | Graph as network input |
| **Lab 5** | Multirobot Path Planning | Capstone project, collaborative learning |

## Key Takeaways

1. **Graphs are ubiquitous** in modern information processing
2. **GNNs enable scalable** machine learning on graph-structured data
3. **Convolutions naturally generalize** from time/space to arbitrary graphs
4. **GNNs = CNNs** with different underlying graph structures
5. **Success requires** understanding and leveraging graph structure

## References

### Course Materials
- Gama, Marques, Leus, Ribeiro (2019). "Convolutional Neural Network Architectures for Signals Supported on Graphs." *IEEE TSP*. [arXiv:1805.00165](https://arxiv.org/abs/1805.00165)
- Ruiz, Gama, Ribeiro (2020). "Graph Neural Networks: Architectures, Stability and Transferability." *Proceedings of the IEEE*. [arXiv:2008.01767](https://arxiv.org/pdf/2008.01767)

### Application Papers
- Segarra et al. (2016). "Attributing the Authorship of the Henry VI Plays by Word Adjacency." *Shakespeare Quarterly*. [DOI: 10.1353/shq.2016.0024](https://doi.org/10.1353/shq.2016.0024)
- Ruiz et al. (2018). "Invariance-Preserving Localized Activation Functions for Graph Neural Networks." [arXiv:1903.12575](https://arxiv.org/abs/1903.12575)
- Tolstaya et al. (2019). "Learning Decentralized Controllers for Robot Swarms with Graph Neural Networks." [arXiv:1903.10527](https://arxiv.org/abs/1903.10527)
- Eisen, Ribeiro (2019). "Optimal Wireless Resource Allocation with Random Edge Graph Neural Networks." [arXiv:1909.01865](https://arxiv.org/abs/1909.01865)

### Lab References
- Huang, Marques, Ribeiro (2018). "Rating Prediction via Graph Signal Processing." *IEEE TSP*. DOI: 10.1109/TSP.2018.2864654
- Li, Gama, Ribeiro, Prorok (2019). "Graph Neural Networks for Decentralized Multi-Robot Path Planning." [arXiv:1912.06095](https://arxiv.org/pdf/1912.06095)

---
