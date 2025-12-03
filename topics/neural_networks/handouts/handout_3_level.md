# Week 00d Handout Level 3: Neural Networks & Deep Learning

## Level 3 Focus
Backpropagation mathematics

## Architectures Covered
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN/LSTM)
- Transformers & Attention



## Backpropagation

Forward: \$a^{(l+1)} = f(W^{(l)}a^{(l)} + b^{(l)})\$

Backward (chain rule):
\$\$\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(l+1)}} \frac{\partial a^{(l+1)}}{\partial W^{(l)}}\$\$

Gradient descent:
\$\$W \leftarrow W - \eta \nabla_W L\$\$

## Applications
- Image classification (CNN)
- Language modeling (Transformer)
- Time series forecasting (LSTM)
