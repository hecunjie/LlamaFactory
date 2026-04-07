import math
import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup


def _select_pause_positions(target_log_probs, m_dit, selection, prob_threshold):
    if len(target_log_probs) == 0 or m_dit <= 0:
        return set()
    if selection == "top_m":
        pairs = sorted(enumerate(target_log_probs), key=lambda x: x[1])
        return {idx for idx, _ in pairs[: min(m_dit, len(pairs))]}
    if selection == "prob_threshold":
        probs = torch.exp(torch.tensor(target_log_probs, dtype=torch.float32)).tolist()
        candidates = [i for i, p in enumerate(probs) if p < prob_threshold]
        if len(candidates) <= m_dit:
            return set(candidates)
        candidates.sort(key=lambda i: probs[i])
        return set(candidates[:m_dit])
    raise ValueError(f"Unknown pause selection mode: {selection}")


def _build_online_paused_batch(
    model,
    batch,
    pause_token_id: int,
    mode: str,
    m_dit: int,
    pause_selection: str,
    pause_prob_threshold: float,
    pad_token_id: int,
    device: str,
    exclude_before_token_ids,
):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    prefix_lens = batch["prefix_lens"]
    target_lens = batch["target_lens"]

    with torch.no_grad():
        score_out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = score_out.logits[:, :-1, :]
        shifted_labels = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_ll = log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)

    new_inputs = []
    new_labels = []
    new_masks = []
    max_new_len = 0
    bsz = input_ids.size(0)

    for b in range(bsz):
        p_len = int(prefix_lens[b].item())
        t_len = int(target_lens[b].item())
        orig_len = p_len + t_len
        seq = input_ids[b, :orig_len].tolist()
        prefix_ids = seq[:p_len]
        target_ids = seq[p_len:]

        start = max(p_len - 1, 0)
        target_ll = token_ll[b, start : start + t_len].tolist()
        selected = _select_pause_positions(target_ll, m_dit, pause_selection, pause_prob_threshold)
        if exclude_before_token_ids:
            selected = {i for i in selected if target_ids[i] not in exclude_before_token_ids}

        cur_in = []
        cur_lb = []
        for tid in prefix_ids:
            cur_in.append(tid)
            cur_lb.append(-100)
        for i, tid in enumerate(target_ids):
            if i in selected:
                cur_in.append(pause_token_id)
                cur_lb.append(pause_token_id if mode == "ditp" else -100)
            cur_in.append(tid)
            cur_lb.append(tid)

        cur_ms = [1] * len(cur_in)
        max_new_len = max(max_new_len, len(cur_in))
        new_inputs.append(cur_in)
        new_labels.append(cur_lb)
        new_masks.append(cur_ms)

    for i in range(bsz):
        pad_len = max_new_len - len(new_inputs[i])
        new_inputs[i] = new_inputs[i] + [pad_token_id] * pad_len
        new_labels[i] = new_labels[i] + [-100] * pad_len
        new_masks[i] = new_masks[i] + [0] * pad_len

    return {
        "input_ids": torch.tensor(new_inputs, dtype=torch.long, device=device),
        "labels": torch.tensor(new_labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(new_masks, dtype=torch.long, device=device),
    }


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
    pause_token_id: Optional[int] = None,
    mode: str = "ditp",
    m_dit: int = 5,
    pause_selection: str = "top_m",
    pause_prob_threshold: float = 0.4,
):
    if pause_token_id is None:
        raise ValueError("pause_token_id is required for online pause insertion.")
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    is_main_process = rank == 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
    exclude_before_token_ids = set()
    if tokenizer is not None:
        if tokenizer.eos_token_id is not None:
            exclude_before_token_ids.add(int(tokenizer.eos_token_id))
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id >= 0 and eot_id != tokenizer.unk_token_id:
            exclude_before_token_ids.add(int(eot_id))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=dataset.collate_fn,
    )
    model = model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = epochs * len(dataloader)
    warmup_steps = math.ceil(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    global_step = 0
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=True,
            dynamic_ncols=True,
            disable=not is_main_process,
        )
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            train_batch = _build_online_paused_batch(
                model=model,
                batch=batch,
                pause_token_id=pause_token_id,
                mode=mode,
                m_dit=m_dit,
                pause_selection=pause_selection,
                pause_prob_threshold=pause_prob_threshold,
                pad_token_id=dataset.pad_token_id,
                device=device,
                exclude_before_token_ids=exclude_before_token_ids,
            )
            outputs = model(**train_batch)
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

            if is_main_process and save_steps > 0 and global_step % save_steps == 0:
                if not save_path:
                    raise ValueError("save_path is required when save_steps > 0")
                ckpt_dir = os.path.join(save_path, f"checkpoint-step-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model_to_save = model.module if isinstance(model, DDP) else model
                model_to_save.save_pretrained(ckpt_dir)
                if tokenizer is not None:
                    tokenizer.save_pretrained(ckpt_dir)
                print(f"Saved checkpoint at step {global_step}: {ckpt_dir}")

        avg_loss = epoch_loss / max(len(dataloader), 1)
        if is_main_process:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

    return model.module if isinstance(model, DDP) else model
