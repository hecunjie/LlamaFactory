"""Phase 3: causal intervention on wrong trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .utils import answers_equal, compute_esoft, find_pre_answer_position, l2_norm, parse_answer


def run_intervention(model, tokenizer, test_trajectories, delta_dict, config, save_path):
    """
    Inject delta vectors at pre-answer position and evaluate correction rate.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(save_path)

    emb_matrix = model.get_input_embeddings().weight.detach()
    dummy_id = _get_dummy_id(tokenizer)
    n_done_questions = 0

    with save_path.open("a", encoding="utf-8") as fout:
        for q in test_trajectories:
            qid = int(q["question_id"])
            question = q["question"]
            system_prompt = q.get("system_prompt")
            user_prompt = q.get("user_prompt")
            prompt_text = q.get("prompt")
            gt = q["ground_truth"]
            if q.get("skipped", False):
                continue

            for wrong_idx, traj in enumerate(q.get("wrong_trajs", [])):
                key = f"{qid}:{wrong_idx}"
                if key in done:
                    continue

                token_ids = traj["token_ids"]
                ids = torch.tensor(token_ids, dtype=torch.long, device=model.device)
                pre_pos = find_pre_answer_position(ids, tokenizer)
                if pre_pos is None:
                    continue
                prefix = ids[: pre_pos + 1]

                h_wrong, e_wrong = _extract_wrong_states(
                    model, prefix, emb_matrix, config.TOP_K_ESOFT
                )
                h_wrong = l2_norm(h_wrong)
                _ = l2_norm(e_wrong)  # Reserved for debugging parity with paper setup.

                baseline = _run_no_injection(
                    model, tokenizer, prefix, dummy_id, gt, config
                )
                results = {}
                for gamma in config.GAMMA_VALUES:
                    gamma_f = float(gamma)
                    h_residual = h_wrong + gamma_f * delta_dict["delta_residual"].to(h_wrong.device)
                    h_vocab = h_wrong + gamma_f * delta_dict["delta_e"].to(h_wrong.device)
                    h_full = h_wrong + gamma_f * delta_dict["delta_h"].to(h_wrong.device)

                    res_out = _run_one_injection(
                        model, tokenizer, prefix, dummy_id, h_residual, gt, config
                    )
                    voc_out = _run_one_injection(
                        model, tokenizer, prefix, dummy_id, h_vocab, gt, config
                    )
                    full_out = _run_one_injection(
                        model, tokenizer, prefix, dummy_id, h_full, gt, config
                    )
                    results[str(gamma)] = {
                        "residual": res_out,
                        "vocab": voc_out,
                        "full": full_out,
                    }

                row = {
                    "question_id": qid,
                    "question": question,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "prompt": prompt_text,
                    "ground_truth": gt,
                    "wrong_traj_index": wrong_idx,
                    "original_wrong_text": traj.get("text"),
                    "baseline": baseline,
                    "results": results,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                done.add(key)

            n_done_questions += 1
            if n_done_questions % 10 == 0:
                print(f"[intervene] processed {n_done_questions} questions")


def _extract_wrong_states(model, prefix_ids, emb_matrix, top_k):
    with torch.no_grad():
        out = model(
            input_ids=prefix_ids.unsqueeze(0),
            output_hidden_states=True,
            return_dict=True,
        )
    last_pos = prefix_ids.shape[0] - 1
    h = out.hidden_states[-1][0, last_pos, :].detach()
    logits = out.logits[0, last_pos, :].detach()
    e = compute_esoft(logits, emb_matrix, top_k=top_k)
    return h, e


def _run_one_injection(model, tokenizer, prefix, dummy_id, h_intervene, ground_truth, config):
    input_ids = torch.cat(
        [prefix, torch.tensor([dummy_id], device=prefix.device, dtype=prefix.dtype)], dim=0
    ).unsqueeze(0)
    done = {"used": False}
    h_inj = h_intervene.detach()

    def _hook(_module, _input, output):
        # Inject only once at the last token embedding of first call.
        if done["used"]:
            return output
        out = output.clone()
        out[:, -1, :] = h_inj
        done["used"] = True
        return out

    handle = model.get_input_embeddings().register_forward_hook(_hook)
    try:
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                do_sample=False,
                max_new_tokens=config.MAX_NEW_TOKENS_INTERVENE,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        handle.remove()

    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    pred = parse_answer(text)
    return {
        "output_text": text,
        "pred_answer": pred,
        "is_correct": answers_equal(pred, ground_truth),
    }


def _run_no_injection(model, tokenizer, prefix, dummy_id, ground_truth, config):
    input_ids = torch.cat(
        [prefix, torch.tensor([dummy_id], device=prefix.device, dtype=prefix.dtype)], dim=0
    ).unsqueeze(0)
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=config.MAX_NEW_TOKENS_INTERVENE,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    pred = parse_answer(text)
    return {
        "output_text": text,
        "pred_answer": pred,
        "is_correct": answers_equal(pred, ground_truth),
    }


def _get_dummy_id(tokenizer):
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    ids = tokenizer.encode(" ", add_special_tokens=False)
    if ids:
        return ids[0]
    return 0


def _load_done(path: Path):
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add(f"{int(row['question_id'])}:{int(row['wrong_traj_index'])}")
    return done
