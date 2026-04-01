import math

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


def train_model(
    model,
    dataset,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_clip: float = 1.0,
    warmup_ratio: float = 0.1,
    device: str = "cuda",
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = epochs * len(dataloader)
    warmup_steps = math.ceil(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")
