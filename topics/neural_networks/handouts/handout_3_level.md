# Neural Networks - Advanced Handout

**Target Audience**: Data scientists and ML engineers
**Duration**: 90 minutes reading
**Level**: Advanced (mathematical foundations, optimization theory)

---

## Mathematical Foundations

### Forward Propagation

For layer $l$ with weights $W^{(l)}$, bias $b^{(l)}$, and activation $f$:

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = f(z^{(l)})$$

Where:
- $a^{(0)} = x$ (input)
- $a^{(L)} = \hat{y}$ (output)

### Backpropagation

**Loss gradient with respect to output**:
$$\delta^{(L)} = \nabla_{a^{(L)}} \mathcal{L} \odot f'(z^{(L)})$$

**Gradient propagation** (for $l = L-1, ..., 1$):
$$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot f'(z^{(l)})$$

**Weight gradients**:
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$$
$$\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}$$

---

## Activation Functions

### ReLU and Variants

**ReLU**: $f(x) = \max(0, x)$, $f'(x) = \mathbf{1}_{x > 0}$

**Leaky ReLU**: $f(x) = \max(\alpha x, x)$ where $\alpha \approx 0.01$

**GELU** (Gaussian Error Linear Unit):
$$f(x) = x \cdot \Phi(x) \approx 0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$$

**Swish**: $f(x) = x \cdot \sigma(\beta x)$

### Softmax for Classification

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

**Numerical stability**:
$$\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^K e^{z_j - \max(z)}}$$

---

## Loss Functions

### Cross-Entropy Loss

**Binary**:
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

**Multi-class**:
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \sum_{c=1}^C y_{ic} \log(\hat{y}_{ic})$$

### Focal Loss (for imbalanced data)
$$\mathcal{L}_{FL} = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

Where $\gamma$ is focusing parameter (typically 2).

---

## Optimization Algorithms

### Stochastic Gradient Descent (SGD)
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t; x_i, y_i)$$

### SGD with Momentum
$$v_t = \gamma v_{t-1} + \eta \nabla_\theta \mathcal{L}$$
$$\theta_{t+1} = \theta_t - v_t$$

### Adam (Adaptive Moment Estimation)
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Default: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

### AdamW (Weight Decay Decoupled)
$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

---

## Initialization Strategies

### Xavier/Glorot Initialization
For layer with $n_{in}$ inputs and $n_{out}$ outputs:

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

**For tanh activation**. Maintains variance through layers.

### He/Kaiming Initialization
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

**For ReLU activation**. Accounts for ReLU zeroing half the activations.

```python
# PyTorch implementation
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.xavier_uniform_(layer.weight)
```

---

## Batch Normalization

### Forward Pass
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

Where $\mu_B$, $\sigma_B^2$ are batch statistics; $\gamma$, $\beta$ are learned.

### Inference
Use running averages instead of batch statistics:
$$\mu_{running} = (1-\alpha)\mu_{running} + \alpha \mu_B$$

### Layer Normalization (for sequences)
Normalize across features instead of batch:
$$\hat{x}_i = \frac{x_i - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}$$

---

## Regularization Theory

### Dropout
During training, randomly zero activations with probability $p$:
$$\tilde{a} = \frac{1}{1-p} \cdot a \cdot m, \quad m_i \sim \text{Bernoulli}(1-p)$$

**Interpretation**: Ensemble of $2^n$ sub-networks.

### L2 Regularization (Weight Decay)
$$\mathcal{L}_{total} = \mathcal{L}_{data} + \frac{\lambda}{2}\|W\|_2^2$$

Gradient becomes:
$$\nabla_W \mathcal{L}_{total} = \nabla_W \mathcal{L}_{data} + \lambda W$$

### Label Smoothing
Instead of one-hot targets, use:
$$y_{smooth} = (1-\alpha)y_{one-hot} + \frac{\alpha}{K}$$

Typically $\alpha = 0.1$.

---

## Attention Mechanism

### Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q \in \mathbb{R}^{n \times d_k}$ (queries)
- $K \in \mathbb{R}^{m \times d_k}$ (keys)
- $V \in \mathbb{R}^{m \times d_v}$ (values)

### Multi-Head Attention
$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O$$
$$head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Self-Attention
When $Q = K = V = X$ (same input):
$$\text{SelfAttention}(X) = \text{softmax}\left(\frac{XX^T}{\sqrt{d}}\right)X$$

---

## Transformer Architecture

### Encoder Block
```
Input -> LayerNorm -> MultiHeadAttention -> Residual -> LayerNorm -> FFN -> Residual -> Output
```

### Position Encoding
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

### Implementation
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x
```

---

## Gradient Issues

### Vanishing Gradients
**Cause**: Saturating activations (sigmoid, tanh) or deep networks.

**Solutions**:
- ReLU activations
- Batch/Layer normalization
- Residual connections
- LSTM/GRU for sequences

### Exploding Gradients
**Cause**: Large weight magnitudes, especially in RNNs.

**Solutions**:
- Gradient clipping: $g \leftarrow \min(1, \frac{\theta}{\|g\|}) \cdot g$
- Weight initialization
- Layer normalization

```python
# Gradient clipping in PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Learning Rate Scheduling

### Cosine Annealing
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))$$

### Warmup + Decay
$$\eta_t = \begin{cases} \eta_{max} \cdot \frac{t}{T_{warmup}} & t < T_{warmup} \\ \eta_{max} \cdot \text{decay}(t - T_{warmup}) & t \geq T_{warmup} \end{cases}$$

```python
# PyTorch scheduler
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

# OneCycleLR (recommended for training)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=len(train_loader)
)
```

---

## Mixed Precision Training

### FP16 Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits**: 2x memory reduction, faster training on modern GPUs.

---

## Distributed Training

### Data Parallel
```python
model = nn.DataParallel(model)  # Simple, single-machine multi-GPU
```

### Distributed Data Parallel
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
model = DDP(model, device_ids=[local_rank])
```

---

## Model Compression

### Quantization
```python
# Post-training quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

### Knowledge Distillation
$$\mathcal{L}_{KD} = \alpha \mathcal{L}_{CE}(y, \hat{y}_{student}) + (1-\alpha) T^2 \mathcal{L}_{KL}(\sigma(z_T/T), \sigma(z_S/T))$$

### Pruning
```python
import torch.nn.utils.prune as prune

# Prune 30% of weights with smallest magnitude
prune.l1_unstructured(module, name='weight', amount=0.3)
```

---

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
2. Vaswani, A., et al. (2017). "Attention Is All You Need"
3. He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
4. Kingma, D. P., & Ba, J. (2015). "Adam: A Method for Stochastic Optimization"

---

*Deep learning is an empirical science. Theory provides guidance, but experimentation determines success.*
