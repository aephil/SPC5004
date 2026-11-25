"""Train the small class-conditional MNIST diffusion model whose samples appear in lec16.

Run with any env that has diffusers, datasets and torchvision:
    python pics/lec16_train.py CKPT_DIR


Deliberately not the lab's setup: MNIST rather than Fashion-MNIST, and a UNet with no attention
blocks so that it trains in minutes on Apple Silicon. Saves the weights to CKPT_DIR and the per-step
losses to CKPT_DIR/losses.txt. Stops after MAX_MINUTES whatever the epoch count.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import DDPMScheduler, UNet2DModel
from torchvision import transforms

OUT = Path(sys.argv[1])
EPOCHS, MAX_MINUTES = 8, 40

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(0)

mnist = load_dataset('ylecun/mnist')['train']
preprocess = transforms.Compose([transforms.ToTensor(), transforms.Pad(2), transforms.Normalize([0.5], [0.5])])


def transform(ex):
    return {'images': [preprocess(im) for im in ex['image']], 'labels': ex['label']}


loader = torch.utils.data.DataLoader(mnist.with_transform(transform), batch_size=128, shuffle=True)

model = UNet2DModel(
    sample_size=32, in_channels=1, out_channels=1, layers_per_block=2,
    block_out_channels=(32, 64, 64),
    down_block_types=('DownBlock2D',) * 3, up_block_types=('UpBlock2D',) * 3,
    num_class_embeds=10,
).to(device)
print(f'parameters: {sum(p.numel() for p in model.parameters()):,}')

scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-5)
losses, start = [], time.time()
for epoch in range(EPOCHS):
    for batch in loader:
        x0 = batch['images'].to(device)
        y = batch['labels'].to(device)
        noise = torch.randn_like(x0)
        t = torch.randint(0, 1000, (x0.shape[0],), device=device)
        pred = model(scheduler.add_noise(x0, noise, t), t, class_labels=y).sample
        loss = F.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    mins = (time.time() - start) / 60
    print(f'epoch {epoch}: loss {sum(losses[-len(loader):]) / len(loader):.4f}, {mins:.1f} min', flush=True)
    model.save_pretrained(OUT)
    if mins > MAX_MINUTES:
        break
(OUT / 'losses.txt').write_text('\n'.join(f'{v:.5f}' for v in losses))
