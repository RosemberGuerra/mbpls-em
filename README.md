# mbpls-em

Probabilistic multi-block PLS with EM estimation.

This package implements a probabilistic latent variable model for integrating multiple datasets (blocks) measured on different samples, decomposing shared and dataset-specific structure, and estimating parameters via an Expectation–Maximization (EM) algorithm.

## Features

- Shared and block-specific latent structure
- EM-based maximum likelihood estimation
- Orthonormal loading updates
- Simulation utilities
- Subspace and reconstruction metrics
- Modular design for research use


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

## Status

Research software. Actively developed.

## Author

Rosember Guerra-Urzola