# Diffusion Models From Scratch

An implementation of **Diffusion Models** - a generative modeling approach that transforms random noise into structured data through an iterative denoising process.

## 🎯 Overview
Diffusion models work by:
1. Gradually adding Gaussian noise to training data
2. Learning to reverse the noise addition process
3. Generating new samples by denoising random noise

## 📚 Structure
- `train_diffusion.py`: Core implementation and training pipeline
    - Noise scheduler
    - U-Net architecture
    - Training loop

## 🔧 Setup & Usage
```bash
# Navigate to directory
cd diffusion_models

# Run training script
python train_diffusion.py
```

## 📝 Documentation
For a detailed explanation of diffusion models and implementation details, check out my [technical blog post](https://pranaval.github.io/Projects/project1.html).