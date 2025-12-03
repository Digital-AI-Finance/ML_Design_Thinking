# Week 00e: Generative AI - The Creation Challenge

## Overview
**Duration**: 90 minutes | **Format**: 4-act narrative | **Slides**: 29 | **Charts**: 20

## Learning Objectives
- Understand generation vs discrimination
- Master VAE latent space representations
- Apply GANs for adversarial training
- Recognize diffusion model principles
- Choose generative method for task
- Navigate quality-diversity tradeoffs

## Structure
1. **Act 1: Challenge** (5 slides) - Why autoencoders fail to generate
2. **Act 2: VAEs** (6 slides) - Variational approach, latent sampling, KL divergence
3. **Act 3: Adversarial & Diffusion** (12 slides) - GANs (forger vs detective), diffusion (reverse corruption)
4. **Act 4: Synthesis** (6 slides) - Modern applications, ethical considerations, method selection

## Key Files
- `act1_challenge.tex` - The generation problem
- `act2_vaes.tex` - Variational autoencoders
- `act3_adversarial_diffusion.tex` - GANs and diffusion models
- `act4_synthesis.tex` - Applications and ethics
- `scripts/create_week0e_charts.py` - Generative architecture diagrams

## Compilation
```powershell
cd Week_00e_Generative_AI
pdflatex 20250928_2200_main.tex  # Run twice
```

## Key Methods
- **VAE**: Smooth latent space, controlled generation
- **GAN**: High quality, mode collapse risk
- **Diffusion**: State-of-art quality, slow sampling
- **Autoregressive**: Sequential generation (GPT-style)

## Modern Applications
- DALL-E / Stable Diffusion (text→image)
- GPT-4 (text generation)
- MusicGen (audio synthesis)
- AlphaFold (protein structure)

## Ethical Considerations
- Deepfakes and misinformation
- Copyright and training data
- Bias amplification
- Environmental impact (compute cost)

## Status
✅ Production Ready - Major overflow reduction, 20 charts verified

## Dependencies
```powershell
pip install torch diffusers transformers matplotlib
```
