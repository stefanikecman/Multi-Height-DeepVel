# Multi-height Tracking of Magnetograms and Dopplergrams using Deep Learning

## Overview

DeepVel 3D Velocity is a PyTorch-based framework for predicting and analyzing horizontal velocity fields from solar data. 
It includes data preparation steps, model training, evaluation, and visualization scripts.

The DeepVel network is originally published in Ramos, A. A., Requerey, I. S., & Vitas, N. (2017). DeepVel: deep learning for the estimation of horizontal velocities at the solar surface. Astronomy & Astrophysics, 604, A11.
https://www.aanda.org/articles/aa/pdf/2017/08/aa30783-17.pdf

## Features

- Neural network models, based on the DeepVel network, for velocity field prediction with variable temporal and spatial input extent
- Hybrid input models with multiple physical inputs (line-of-sight magnetic field, line-of-sight velocity)
- Training on data with different cadence and different optical depths
- Data normalization/denormalization codes
- Evaluation metrics: MSE, RMSE, Pearson correlation (vx and vy), slope (vy and vy) for the velocities, divergence and vorticity
- Visualization: prediction maps, scatter plots, plots with visualized arrows

## Getting Started

### Prerequisites

- Python 3.10
- PyTorch (with CUDA for GPU support)

### Installation

```bash
git clone <repo_url>
cd Multi-Height-DeepVel
conda create --name <env-name>
conda activate <env-name>
pip install -r requirements.txt
```

### Data Preparation

Methods for data preparation, normalization and other preprocessing steps are located in `prepare_data.py`.

### Training
Single-input DeepVel model definition is in `deepvel_torch_benchmarking.py`

Train a model using:

```bash
python deepvel_torch_benchmarking.py
```
Hybrid-input DeepVel model variants are defined in `hybrid_deepvel_torch.py`

Train a model using:

```bash
python hybrid_deepvel_torch.py
```

### Evaluation & Visualization

Evaluate models and generate plots (first uncomment the chosen evaluation and adjust the paths):

```bash
python test_benchmark.py
```
or

```bash
python test_multiheight_hybrid_models.py
```

## Data
This project uses simulation data generated with the MURaM  code, courtesy of Matthias Rempel (High Altitude Observatory).
All intensity, velocity and magnetic field data used in this project is based on MURaM simulations.
Data is located on the KIS servers and can be obtained upon request.
