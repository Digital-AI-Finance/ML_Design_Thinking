# Week 00d: Neural Networks - The Depth Challenge

## Overview
**Duration**: 90 minutes | **Format**: 4-act narrative | **Slides**: 27 | **Charts**: 25

## Learning Objectives
- Understand why shallow networks fail
- Master backpropagation and gradient flow
- Apply CNNs for visual recognition
- Recognize attention mechanisms and transformers
- Choose architecture for data type
- Navigate depth vs width tradeoffs

## Structure
1. **Act 1: Challenge** (5 slides) - XOR problem, perceptron limitations
2. **Act 2: Shallow MLPs** (6 slides) - Hidden layers, universal approximation, vanishing gradients
3. **Act 3: Modern Architectures** (10 slides) - CNNs, RNNs, Transformers, attention
4. **Act 4: Synthesis** (6 slides) - Architecture selection, inductive bias, applications

## Key Files
- `act1_challenge.tex` - The perceptron failure
- `act2_shallow_mlps.tex` - Multi-layer networks
- `act3_modern_architectures.tex` - CNN/RNN/Transformer deep dive
- `act4_synthesis.tex` - When to use what
- `scripts/create_week0d_charts.py` - Neural architecture diagrams

## Compilation
```powershell
cd Week_00d_Neural_Networks
pdflatex 20251007_1700_neural_networks.tex  # Run twice
```

## Key Architectures
- MLP: Fully connected, tabular data
- CNN: Convolutional, spatial data (images)
- RNN/LSTM: Recurrent, sequential data (time series)
- Transformer: Attention, language/sequences

## Status
✅ Production Ready - Unicode compliant, mathematically verified

## Dependencies
```powershell
pip install torch torchvision transformers matplotlib
```
