# HyDIN

This repository contains the implementation of **HyDIN** for protein-protein interaction (PPI) prediction. According to the accompanying paper, the core idea is to combine a **Lorentz-space hyperbolic encoder** with a **multi-scale diffusion power module** so that hierarchical PPI topology and cross-scale interaction patterns can be modeled more naturally.


## Default Task

- Input the protein feature matrix `protein_feature.txt`
- Input the protein-protein interaction adjacency matrix `mat_protein_protein.txt`
- Predict whether an interaction exists between previously unknown protein pairs

## Main Features

- Hyperbolic structure encoding: `HyboNet` embeds node features in the Lorentz manifold to better match the hierarchical organization of PPI networks described in the paper
- Multi-scale diffusion fusion: `GWTNet` in `models/models.py` propagates features across multiple diffusion scales and fuses them with gated weighting
- Contrastive-learning utilities: `train.py` includes stochastic feature/graph augmentation and a hyperbolic contrastive objective for robustness under perturbation
- Link prediction evaluation: supports ROC-AUC, AP
- Bootstrap statistics: evaluation computes confidence intervals and p-values
- Two training modes:
  - single split training
  - K-fold cross-validation

## Directory Structure

```text
HyDIN/
|-- data/                  data files
|   |-- protein_feature.txt
|   |-- mat_protein_protein.txt
|   |-- drug_feature.txt
|   |-- disease_feature.txt
|   |-- mat_drug_drug.txt
|   |-- mat_drug_protein.txt
|   `-- zeng/              backup / reference data
|-- layers/                graph layers and decoders
|-- manifolds/             hyperbolic manifold implementations
|-- models/                encoders and diffusion / wavelet models
|-- utils/                 data processing, training, and evaluation utilities
|-- train.py               training entry point
`-- requirements.txt       dependency list
```



## Data Sources and Reference Datasets

The current data organization is related to the following public repositories:

- DTINet (Luo dataset)
  - repository: <https://github.com/luoyunan/DTINet>
  - description: heterogeneous network data and code for drug-target interaction prediction
  - the public data include drugs, proteins, diseases, side effects, and multiple association matrices such as `mat_drug_drug.txt`, `mat_drug_protein.txt`, and `mat_protein_protein.txt`

- deepDTnet (Zeng dataset)
  - repository: <https://github.com/ChengF-Lab/deepDTnet>
  - description: heterogeneous network deep learning data and code for known drug-target identification
  - the public data include files such as `proteinprotein.txt` and `proteinsim1network.txt` to `proteinsim4network.txt`

If you want to replace the default data, the minimum requirement is that the input files keep the following format:

- `protein_feature.txt`: a plain-text matrix with one node per row and one feature per column
- `mat_protein_protein.txt`: a plain-text adjacency matrix, square, with entries usually stored as `0/1`

## Environment Requirements

The core dependencies listed in `requirements.txt` are:

```txt
numpy==1.16.2
scikit-learn==0.20.3
torch
geoopt==0.5.0
torchvision==0.2.2
networkx==2.2
matplotlib<3.3
```

## Installation

```bash
pip install -r requirements.txt
```

If multiple Python versions are installed in your environment, it is safer to specify the interpreter explicitly.

## Running

Run the training script directly:

```bash
python train.py
```



## Custom Data

If you want to switch the project to your own PPI dataset, the minimum-change workflow is usually:

1. Replace `data/protein_feature.txt` with your own node features
2. Replace `data/mat_protein_protein.txt` with your own adjacency matrix
3. Make sure the number of nodes is consistent
4. Make sure the feature matrix and adjacency matrix follow the same node ordering

If you want to introduce drugs, diseases, or other multimodal relations, the current default training path is not sufficient. You will need to continue modifying `load_data_lp()` and the downstream training logic.
