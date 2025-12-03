# Week 00e Handout Level 3: Generative AI

## Level 3 Focus
VAE/GAN/Diffusion mathematics

## Generative Models
- Variational Autoencoders (VAE)
- Generative Adversarial Networks (GAN)
- Diffusion Models (DDPM, Stable Diffusion)
- Large Language Models (GPT, LLaMA)



## VAE Loss
\$\$L = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x)\|p(z))\$\$

Reconstruction + KL divergence

## GAN Minimax
\$\$\min_G \max_D \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1-D(G(z)))]\$\$

## Diffusion Forward
\$\$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)\$\$

## Ethical Considerations
- Deepfakes and misinformation
- Copyright and training data
- Bias amplification
- Computational cost
