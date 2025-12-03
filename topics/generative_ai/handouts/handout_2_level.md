# Week 00e Handout Level 2: Generative AI

## Level 2 Focus
Using Stable Diffusion, GPT APIs

## Generative Models
- Variational Autoencoders (VAE)
- Generative Adversarial Networks (GAN)
- Diffusion Models (DDPM, Stable Diffusion)
- Large Language Models (GPT, LLaMA)

## Using Stable Diffusion

\`\`\`python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "A serene mountain landscape at sunset"
image = pipe(prompt).images[0]
image.save("output.png")
\`\`\`

## GPT API
\`\`\`python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain ML"}]
)
print(response.choices[0].message.content)
\`\`\`



## Ethical Considerations
- Deepfakes and misinformation
- Copyright and training data
- Bias amplification
- Computational cost
