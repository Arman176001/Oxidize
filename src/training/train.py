import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from src.model import EDTransformer, ModelArgs
from dataset import YourCustomDataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
import math
import os

# -----------------------------------------------------------------------------
# This Demo Training loop inspired by DeepMind's AlphaCode (Li et al., 2022)
# Implements key training strategies from the paper:
# - AdamW optimizer with β1=0.9, β2={0.999 or 0.95}, weight decay=0.1
# - Learning rate warmup from 1e-9 to 1e-4 over 1000 steps
# - Cosine learning rate decay to 1e-5
# - Global gradient norm clipping at 1.0
# Reference: https://arxiv.org/pdf/2203.07814
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

if dist.is_available() and dist.is_initialized():
    dist.init_process_group(backend='nccl')

args = ModelArgs()
epochs = 10
batch_size = 32
grad_accum_steps = 2
use_amp = True
warmup_steps = 1000
total_training_steps = epochs * (len(YourCustomDataset(split="train")) // batch_size)

initial_lr = 1e-4
final_lr = 1e-5
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.999  # or 0.95 for larger models

train_dataset = YourCustomDataset(split="train")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = EDTransformer(args).to(device)
optimizer = AdamW(model.parameters(), lr=initial_lr, betas=(beta1, beta2), weight_decay=weight_decay)
scaler = GradScaler('cuda',enabled=use_amp)
criterion = nn.CrossEntropyLoss()

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps)) * (initial_lr / 1e-9)
    progress = (current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
    cosine_lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + math.cos(math.pi * progress))
    return cosine_lr / initial_lr

scheduler = LambdaLR(optimizer, lr_lambda)

model.train()
global_step = 0

for epoch in range(epochs):
    total_loss = 0
    for step, batch in enumerate(train_loader):
        encoder_tokens, decoder_input, decoder_target = [x.to(device) for x in batch]

        with autocast(device_type='cuda', enabled=use_amp):
            logits = model(encoder_tokens, decoder_input)
            logits = logits.view(-1, logits.size(-1))
            loss = criterion(logits, decoder_target.view(-1))

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            scheduler.step()
            global_step += 1

        total_loss += loss.item()

        if (step + 1) % 50 == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}], Loss: {loss.item():.4f}, LR: {current_lr:.6e}")

    if (not dist.is_initialized() or dist.get_rank() == 0):
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")

if (not dist.is_initialized() or dist.get_rank() == 0):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/edtransformer.pt")
