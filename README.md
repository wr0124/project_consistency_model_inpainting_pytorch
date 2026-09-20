# Consistency Models for Image Inpainting in PyTorch

Experimental PyTorch implementation of **Improved Consistency Training** and **consistency-based sampling/editing** for image generation and inpainting.

The project explores how consistency models can be trained and used for masked image reconstruction with a U-Net architecture.

#data download

!gdown 1FnzQLDPs-IlTTEr14YyENKjTYqZfn8mS  && tar -xf butterflies256.tar.gz # Butterflies Dataset

!gdown 1VJow74U3H7KG_HOiP1WWo6LoqoE3azJj && tar -xf anime_faces.tar.gz # Anime Faces

## Overview

The implementation includes:

- U-Net-based consistency model
- Improved Consistency Training
- Karras noise schedule
- log-normal timestep sampling
- pseudo-Huber consistency loss
- consistency-model output scaling and skip connections
- mask-based image inpainting
- multi-step consistency sampling
- checkpoint saving/loading
- training and sample visualization with Visdom

## Method

The training pipeline follows the Improved Consistency Training approach.

During training:

- images are corrupted at sampled noise levels
- a random mask is generated for inpainting experiments
- noisy inputs are passed through the consistency model
- predictions at adjacent noise levels are compared
- the loss is weighted according to the noise schedule

The implementation uses a pseudo-Huber loss for consistency training.

## Inpainting and sampling

`ConsistencySamplingAndEditing` implements iterative sampling and masked editing.

The inpainting process preserves the unmasked region while progressively reconstructing the masked area across a sequence of noise levels.

Example sampling schedule:

```text
80.0 → 24.4 → 5.84 → 0.9 → 0.661
```

The code also supports:

- one-step and multi-step sampling
- clipping generated values
- custom masks
- transformation and inverse-transformation functions

## Repository structure

```text
.
├── UNet.py                    # U-Net architecture
├── consistency_generator.py  # Improved Consistency Training implementation
├── inference.py               # Sampling and inpainting
├── trainv2.py                 # Main experimental training script
├── train.py                   # Earlier training implementation
├── options.py                 # Command-line configuration
├── utils.py                   # Utility functions
└── run_model.sh               # Example training command
```

## Technologies

Python · PyTorch · TorchVision · Consistency Models · Generative AI · U-Net · Image Inpainting · Computer Vision · Visdom

### Installation

```bash
pip install -r requirements.txt

## Running an experiment

The example configuration is available in:

```bash
run_model.sh
```

Before running the experiment, update the dataset path and other parameters for your environment.

Example:

```bash
python3 trainv2.py \
  --data_dir /path/to/image/dataset \
  --image_size 32 32 \
  --batch_size 32 \
  --num_workers 4 \
  --max_steps 100000 \
  --sample_every_n_steps 500 \
  --lr 1e-4 \
  --num_samples 32 \
  --env consistency_model
```

The dataset is expected to follow the directory structure supported by `torchvision.datasets.ImageFolder`.

## Training configuration

The command-line interface exposes parameters including:

```text
--data_dir
--image_size
--batch_size
--num_workers
--max_steps
--sample_every_n_steps
--lr
--lr_scheduler_start_factor
--lr_scheduler_iters
--sampling_sigmas
--num_samples
--env
```

## Visualization

Training losses, input images, intermediate predictions, and generated samples are visualized using **Visdom**.

## Status

This repository contains **research and experimental code** for studying consistency models and image inpainting. It is not intended as a production-ready library.

## Author

**Ru Wang Pujos**

Machine Learning R&D Engineer  
Computer Vision · Deep Learning · Generative AI

Portfolio: https://wr0124.github.io/
