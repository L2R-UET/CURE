# CURE: Counterfactual Understanding via Retrieval-aware Multimodal Modeling for Time-to-Event Survival Prediction
This repository contains the official implementation of the paper:
> **Counterfactual Understanding via Retrieval-aware Multimodal Modeling for Time-to-Event Survival Prediction**  
> *Submitted to ECIR 2026 (IR for Good Track, 48th European Conference on Information Retrieval)*
---
## Table of Contents
- Overview
- Key Features
- Setup
- Data Preparation
- Set up
---
## Overview
This project introduces CURE, a retrieval-aware multimodal framework for counterfactual survival prediction.
It integrates omics, clinical, and paraclinical modalities through a latent retrieval mechanism that models treatment–response dynamics beyond raw feature similarity.
CURE builds upon several strong research foundations:
- Base Architecture: Extends DeepCoxMixtures-HeterogenousEffects (CMHE) for heterogeneous treatment phenotypes.
- Fusion Mechanism: Inspired by Multimodal Transformer (MulT), enabling cross-modal alignment.
- Representation Learning: Uses a Mixture-of-Experts Autoencoder (MoE-AE) for adaptive modality compression.
---
## Key Features
- Retrieval-aware Counterfactual Reasoning: Learns latent subpopulation priors to infer individualized treatment effects.
- Multimodal Integration: Combines omics (mRNA, CNA, Methylation, Mutation) with clinical and demographic features.
- Transformer-based Fusion: Utilizes pairwise cross-modal attention to align heterogeneous modalities.
- Adaptive Mixture-of-Experts (MoE-AE): Enhances representation diversity and prevents overfitting in high-dimensional omics data.
- Survival Estimation via CMHE: Models both baseline and treatment-specific hazard dynamics.
---
## Data Preparation 
### Download the Datasets
All datasets used in this work are publicly available via https://www.cbioportal.org/
 — a comprehensive cancer genomics platform hosting TCGA and METABRIC studies.
 | Dataset       | Description                                                | cBioPortal Study Link                                                                       |
| ------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **METABRIC**  | Breast Cancer (mRNA, CNA, Methylation, Mutation, Clinical) | [METABRIC (Breast Cancer)](https://www.cbioportal.org/study/summary?id=brca_metabric)       |
| **TCGA-LUAD** | Lung Adenocarcinoma                                        | [TCGA-LUAD Study](https://www.cbioportal.org/study/summary?id=luad_tcga_pan_can_atlas_2018) |
### Dictionary Structure
After downloading and extracting, place the datasets in the datasets/ directory following the structure below:
```bash
CURE/
 ├── datasets/
 │   ├── metabric/
 │   │   ├── ...
 │   │
 │   └── tcga-luad/
 │       ├── ...
 │
 └── src/
     ├── cross_modal.py
     ├── data.py
     ├── experiment.py
     ├── init.py
     └── ...
```
---
## Set up
Clone the repository and install dependencies:
```bash
git clone https://github.com/L2R-UET/MAMEX.git
cd CURE
pip install -r requirements.txt
```
Run main training pipeline:
```bash
python src/main.py
```
