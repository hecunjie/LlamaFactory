import math
import os
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
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
    save_steps: int = 0,
    save_path: Optional[str] = None,
    tokenizer=None,
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

    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=True,
            dynamic_ncols=True,
        )
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Encountered non-finite loss (NaN/Inf). "
                    "Try smaller lr, bf16/fp32, or verify data quality."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            loss_val = loss.item()
            epoch_loss += loss_val
            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                step=global_step,
            )

            if save_steps > 0 and global_step % save_steps == 0:
                if not save_path:
                    raise ValueError("save_path is required when save_steps > 0")
                ckpt_dir = os.path.join(save_path, f"checkpoint-step-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                if tokenizer is not None:
                    tokenizer.save_pretrained(ckpt_dir)
                print(f"Saved checkpoint at step {global_step}: {ckpt_dir}")

        avg_loss = epoch_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")
