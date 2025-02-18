# Mamba

An implementation of MAMBA - a selective state space sequence model, a strong competitor of Transformers. The implementation is adapted from:

- [mamba-minimal](https://github.com/johnma2006/mamba-minimal)
- [mamba-tiny](https://github.com/PeaBrane/mamba-tiny/tree/master)

The Mamba model is a sequence modeling architecture that utilizes selective scanning, convolutional layers, and residual connections.

## Overview

- Uses `MambaBlock` for sequence processing
- Implements `SelectiveScan` for efficient state-space computation
- Includes training code with a dummy dataset

## Model Architecture

The Mamba model consists of:

- **RMSNorm**: Root Mean Square normalization
- **LocalConv**: Depthwise convolutional layer
- **SelectiveScan**: State-space computation module
- **MambaBlock**: The main processing block combining convolution and scanning
- **ResidualBlock**: Applies residual connections with normalization
- **MambaModel**: The complete model with embedding, layers, and language modeling head

## Setup & Usage
```bash
# Navigate to directory
cd mamba

# Run training script
python run_mamba.py
```

## 📝 Documentation
For a detailed explanation of Mamba and implementation details, check out my [technical blog post](https://pranaval.github.io/Projects/project2.html).


