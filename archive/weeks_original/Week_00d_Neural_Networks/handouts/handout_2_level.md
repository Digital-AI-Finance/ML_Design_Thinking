# Week 00d Handout Level 2: Neural Networks & Deep Learning

## Level 2 Focus
PyTorch/TensorFlow implementation

## Architectures Covered
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN/LSTM)
- Transformers & Attention

## PyTorch Example

\`\`\`python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
\`\`\`



## Applications
- Image classification (CNN)
- Language modeling (Transformer)
- Time series forecasting (LSTM)
