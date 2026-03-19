"""Phase 1: sample trajectories and split into correct/wrong groups."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .utils import answers_equal, parse_answer


SYSTEM_PROMPT = (
    'You are a helpful math assistant. Please solve the problem step by step '
    'and put the final answer after "The answer is ".'
)


def collect_trajectories(model, tokenizer, dataset_split, config, save_path):
    """
    Sample multiple trajectories for each question and store grouped results.
    Supports checkpoint-resume by question_id.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_existing(save_path)
    done_ids = {int(x["question_id"]) for x in data}

    for qid, ex in enumerate(dataset_split):
        if qid in done_ids:
            continue

        question = ex["question"].strip()
        ground_truth_raw = ex["answer"].strip()
        ground_truth = parse_answer(ground_truth_raw)
        prompt_text = _build_prompt(tokenizer, question)

        correct_trajs = []
        wrong_trajs = []

        for _ in range(config.N_TRAJECTORIES):
            generated_text, token_ids = _sample_once(model, tokenizer, prompt_text, config)
            pred = parse_answer(generated_text)
            item = {
                "text": generated_text,
                "token_ids": token_ids,
                "predicted_answer": pred,
            }
            if answers_equal(pred, ground_truth):
                correct_trajs.append(item)
            else:
                wrong_trajs.append(item)

        skipped = (
            len(correct_trajs) < config.MIN_TRAJS_PER_GROUP
            or len(wrong_trajs) < config.MIN_TRAJS_PER_GROUP
        )
        data.append(
            {
                "question_id": qid,
                "question": question,
                "ground_truth": ground_truth,
                "correct_trajs": correct_trajs,
                "wrong_trajs": wrong_trajs,
                "skipped": skipped,
            }
        )
        _save_json(save_path, data)

        if (qid + 1) % 10 == 0:
            print(f"[collect] processed {qid + 1} questions")

    return data


def _sample_once(model, tokenizer, prompt_text: str, config):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            max_new_tokens=config.MAX_NEW_TOKENS_COLLECT,
        )
    out_ids = outputs[0]
    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return text, out_ids.tolist()


def _build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _load_existing(path: Path):
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
