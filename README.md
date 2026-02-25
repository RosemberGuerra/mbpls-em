![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

# mbpls-em

Probabilistic multi-block PLS with EM estimation in Python.

This package implements a probabilistic latent variable model for integrating multiple datasets (blocks) measured on different samples, decomposing shared and dataset-specific structure, and estimating parameters via an Expectation–Maximization (EM) algorithm.
<img width="888" height="571" alt="Probabilistic multi-block PLS with EM estimation  - visual selection" src="https://github.com/user-attachments/assets/8597303c-392f-4150-a930-ff74eb5a42f5" />

## Features
- Shared and block-specific latent structure
- EM-based maximum likelihood estimation
- Orthonormal loading updates
- Simulated data utilities
- Subspace and reconstruction metrics
- Modular design for research use

## Mathematical Model Structure
We consider $K$ datasets $(X_k, Y_k)$, with $X_k\in \mathbb{R}^{N_k \times d}$ and an associated response $Y_k\in \mathbb{R}^{N_k}$, for $k=1,\dots,K$. Our goal is to integrate the $K$ datasets while model the (response) variables $Y_k$. To achieve this, we introduce the following latent variable model
$$
X_k &= T_k W^\top + U_k P_k^\top + E_k,
$$

$$
Y_k &= T_k \beta_k^\top + U_k \phi_k^\top + \varepsilon_k.
$$

where the shared loading structure among the datasets $X_k$ is represent by $W$ while the specific loading structure is represented by $P_k$. The share latent scores between $X_k$ and $Y_k$ is $T_k\in \mathbb{R}^{N_k \times r}$ and $U_k \in \mathbb{R}^{N_k \times q_k}$. $E_k$ and $\varepsilon_k$ denote the noises.

## Conceptual Backgroud
This work was conceptually inspired by:

Said el Bouhaddani et al., PLOS Computational Biology (2024)
https://doi.org/10.1371/journal.pcbi.1011809

However, **this repository is not a reimplementation of that work**.

Important distinctions:

- The model formulation differs.
- The EM estimation routine was independently derived.
- The implementation is fully written in Python.
- No R code from the original repository was reused or translated.

This repository represents an independent methodological and computational contribution.

## Installation

### Install directly from GitHub

```bash
pip install git+https://github.com/RosemberGuerra/mbpls-em.git
```


### Clone the repository (Development install):

```bash
git clone https://github.com/RosemberGuerra/mbpls-em.git
cd mbpls-em
pip install -e .
```
## Example Usage
(To be added)
## Status

Research software. Actively developed.

## Author

Rosember Guerra-Urzola

## Funding
Developed as part of  *Multi-omics approach to predict therapeutic targets for multiple system atrophy*, under Grant No. 418452464.
