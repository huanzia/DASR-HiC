# DASR-HiC

DASR-HiC is a deep-learning-based framework for Hi-C contact map super-resolution, focusing on the recovery of long-range chromatin interactions. The project is built on a DiCARN-style reconstruction backbone and introduces distance-aware supervision to strengthen the learning of off-diagonal and far-distance interaction signals.

## Overview

High-resolution Hi-C data are useful for studying fine-scale chromatin organization and long-range regulatory interactions, but generating deeply sequenced Hi-C maps is expensive and technically demanding.

Most existing Hi-C super-resolution methods treat contact-map reconstruction as a general image restoration task and optimize all matrix entries with a globally uniform loss. However, Hi-C matrices have clear distance-dependent structures: near-diagonal contacts are usually dense and easier to recover, while long-range contacts are sparse, weak, and biologically informative.

DASR-HiC addresses this issue by incorporating genomic-distance information into the reconstruction objective. The model assigns stronger supervision to off-diagonal and far-distance contacts, reducing the dominance of dense near-diagonal signals during training.
[View overview.pdf](./overview.pdf)

## Method

DASR-HiC contains two main components:

1. **DiCARN-style reconstruction backbone**
   - Reconstructs high-resolution Hi-C patches from low-resolution inputs.
   - Uses cascading residual and dense feature transformation modules.
   - Keeps the reconstruction architecture stable for controlled comparison.

2. **Distance-aware supervision**
   - Assigns different reconstruction weights according to genomic distance.
   - Increases the contribution of long-range contact errors during training.
   - Encourages the model to recover structurally informative off-diagonal regions.

Two distance-aware variants are included:

- **DistOnly**: a moderate distance-aware weighting strategy, mainly used for cross-cell-line generalization.
- **DistFarStrong**: a stronger far-distance weighting strategy, mainly used for structure-oriented long-range contact recovery.

The total objective is:

$$
L_{total} = L_{rec} + \lambda_{dist}L_{dist}
$$

where `L_rec` is the standard reconstruction loss, `L_dist` is the distance-aware weighted reconstruction loss, and `lambda_dist` controls the contribution of the distance-aware term.

## Repository Structure

```text
DASR-HiC/
├── Models/                 # Model definitions
├── Utils/                  # Data processing and metric utilities
├── DASR-HIC_Train.py       # Training entry
├── Predict.py              # Prediction entry
├── Read_Data.py            # Hi-C data reading and preprocessing
├── Generate.py             # Data generation script
├── Downsample.py           # Low-resolution data generation script
├── DNase_Imputation.R      # DNase-related processing script
├── environment.yml         # Conda environment configuration
├── overview.pdf            # Method overview
└── README.md               # Project documentation
```

## Environment Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate dasr-hic
```

If the environment name in `environment.yml` is different, use the name defined in the file.

## Data Preparation

The project requires processed Hi-C contact map data as model input. The data should be prepared as low-resolution and high-resolution contact-map patches before training.

The general preprocessing procedure includes:

1. Read Hi-C contact matrices.
2. Generate high-resolution target patches.
3. Generate low-resolution input patches by downsampling.
4. Split the data into training, validation, and testing sets.
5. Place the processed files in the corresponding local data directory.

Related scripts:

```bash
python Read_Data.py
python Downsample.py
python Generate.py
```

Before running these scripts, modify the input and output paths according to your local data location.

## Training

Run DistFarStrong training from the project root directory:

```bash
python DASR-HIC_Train.py \
  -c GM12878 \
  -hr 10kb \
  -lr 40kb \
  --loss_mode dist_far_strong \
  --epochs 100 \
  --batch_size 16 \
  --base_lr 2e-4 \
  --lambda_dist 0.5
```

Run DistOnly training from the project root directory:

```bash
python DASR-HIC_Train.py \
  -c GM12878 \
  -hr 10kb \
  -lr 40kb \
  --loss_mode dist_only \
  --epochs 100 \
  --batch_size 16 \
  --base_lr 2e-4 \
  --lambda_dist 0.5
```

Before training, check the following configurations:

- training data path
- validation data path
- model save path
- batch size
- learning rate
- number of epochs
- loss variant
- distance-aware weight setting
- `lambda_dist`

Supported training variants include:

```text
dist_only
dist_far_strong
```

## Prediction

After training, run the prediction script:

```bash
python Predict.py
```

Before prediction, check the following configurations:

- test data path
- checkpoint path
- output path
- prediction save format

## Contact

Yu Chen: nefu_chenyu@nefu.edu.cn
