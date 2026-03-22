"""Phase 2: compute delta vectors from trajectory groups."""

from __future__ import annotations

import torch

from .utils import compute_esoft, find_pre_answer_position, l2_norm, model_input_device


def compute_delta(model, tokenizer, trajectories_data, config, save_path):
    """
    Compute delta_h, delta_e and delta_residual from collected trajectories.
    """
    emb = model.get_input_embeddings().weight.detach()
    correct_h, wrong_h = [], []
    correct_e, wrong_e = [], []

    for q in trajectories_data:
        if q.get("skipped", False):
            continue
        for traj in q.get("correct_trajs", []):
            h, e = _extract_h_and_esoft(model, tokenizer, traj["token_ids"], emb, config)
            if h is not None:
                correct_h.append(l2_norm(h))
                correct_e.append(l2_norm(e))
        for traj in q.get("wrong_trajs", []):
            h, e = _extract_h_and_esoft(model, tokenizer, traj["token_ids"], emb, config)
            if h is not None:
                wrong_h.append(l2_norm(h))
                wrong_e.append(l2_norm(e))

    if not correct_h or not wrong_h:
        raise RuntimeError("Not enough valid trajectories to compute delta.")

    h_correct_mean = torch.stack(correct_h, dim=0).mean(dim=0)
    h_wrong_mean = torch.stack(wrong_h, dim=0).mean(dim=0)
    e_correct_mean = torch.stack(correct_e, dim=0).mean(dim=0)
    e_wrong_mean = torch.stack(wrong_e, dim=0).mean(dim=0)

    delta_h = h_correct_mean - h_wrong_mean
    delta_e = e_correct_mean - e_wrong_mean
    delta_residual = delta_h - delta_e

    out = {
        "delta_h": delta_h,
        "delta_e": delta_e,
        "delta_residual": delta_residual,
        "n_correct": len(correct_h),
        "n_wrong": len(wrong_h),
        "hidden_dim": int(delta_h.shape[0]),
    }
    torch.save(out, save_path)

    print(f"[delta] ||delta_h||={torch.norm(delta_h).item():.6f}")
    print(f"[delta] ||delta_e||={torch.norm(delta_e).item():.6f}")
    print(f"[delta] ||delta_residual||={torch.norm(delta_residual).item():.6f}")
    return out


def _extract_h_and_esoft(model, tokenizer, token_ids, embedding_matrix, config):
    ids = torch.tensor(token_ids, dtype=torch.long, device=model_input_device(model))
    if ids.numel() < 2:
        return None, None
    pre_pos = find_pre_answer_position(ids, tokenizer)
    # Fallback: if phrase is missing, use penultimate token position.
    if pre_pos is None or pre_pos < 0 or pre_pos >= ids.shape[0]:
        pre_pos = ids.shape[0] - 2
        if pre_pos < 0:
            return None, None

    with torch.no_grad():
        out = model(
            input_ids=ids.unsqueeze(0),
            output_hidden_states=True,
            return_dict=True,
        )
    hidden = out.hidden_states[config.HIDDEN_LAYER_INDEX][0, pre_pos, :].detach()
    logits = out.logits[0, pre_pos, :].detach()
    esoft = compute_esoft(logits, embedding_matrix, top_k=config.TOP_K_ESOFT)
    return hidden, esoft
