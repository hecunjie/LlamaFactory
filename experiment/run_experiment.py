"""Run all 4 phases for hidden-state delta intervention experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from experiment import config as default_config
    from experiment.analysis import analyze_and_plot
    from experiment.collect_trajectories import collect_trajectories
    from experiment.compute_delta import compute_delta
    from experiment.intervention import run_intervention
    from experiment.utils import (
        answers_equal,
        model_input_device,
        parse_answer,
        split_system_user_from_prompt,
    )
else:
    from . import config as default_config
    from .analysis import analyze_and_plot
    from .collect_trajectories import collect_trajectories
    from .compute_delta import compute_delta
    from .intervention import run_intervention
    from .utils import (
        answers_equal,
        model_input_device,
        parse_answer,
        split_system_user_from_prompt,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--delta_source", type=str, default=None, choices=["train", "test"])
    parser.add_argument("--skip_collect", action="store_true")
    parser.add_argument("--skip_delta", action="store_true")
    parser.add_argument("--skip_intervene", action="store_true")
    parser.add_argument("--n_questions", type=int, default=None)
    parser.add_argument("--train_sampled_jsonl", type=str, default=None)
    parser.add_argument("--test_sampled_jsonl", type=str, default=None)
    parser.add_argument(
        "--test_dataset",
        type=str,
        default=None,
        help="Local test dataset path (json/jsonl) with question+answer (or instruction+output)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        choices=["auto", "single"],
        help="auto: multi-GPU via HF device_map; single: one GPU .to(cuda). Default: config DEVICE_MAP",
    )
    args = parser.parse_args()

    cfg = _build_runtime_config(args)
    save_dir = Path(cfg.SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("[run] loading tokenizer/model ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.DTYPE == "bfloat16" else torch.float16
    map_mode = args.device_map or getattr(cfg, "DEVICE_MAP", "auto")
    use_auto_map = (
        cfg.DEVICE == "cuda"
        and torch.cuda.is_available()
        and map_mode == "auto"
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if use_auto_map else None,
        low_cpu_mem_usage=use_auto_map,
    ).eval()
    if not use_auto_map:
        if cfg.DEVICE == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        else:
            model = model.to("cpu")
    print(
        f"[run] model load: device_map={'auto' if use_auto_map else 'None (single/cpu)'}, "
        f"input_device={model_input_device(model)}"
    )

    ds_train = None
    ds_test = None
    if not args.skip_collect:
        if args.test_dataset:
            print(f"[run] loading local test dataset: {args.test_dataset}")
            ds_test = _load_local_qa_dataset(args.test_dataset, n_questions=args.n_questions)
        else:
            print("[run] loading datasets from HF ...")
            ds_train = load_dataset(cfg.DATASET_NAME, cfg.DATASET_CONFIG, split="train")
            ds_test = load_dataset(cfg.DATASET_NAME, cfg.DATASET_CONFIG, split="test")
            if args.n_questions is not None:
                ds_train = ds_train.select(range(min(args.n_questions, len(ds_train))))
                ds_test = ds_test.select(range(min(args.n_questions, len(ds_test))))

    test_traj_path = save_dir / "test_trajectories.json"
    train_traj_path = save_dir / "train_trajectories.json"
    delta_path = save_dir / "delta.pt"
    inter_path = save_dir / "intervention_results.jsonl"

    # Phase 1
    if not args.skip_collect:
        if ds_test is None:
            raise RuntimeError("Dataset is required for collection stage.")
        print("[run] phase1 collect(test)")
        collect_trajectories(model, tokenizer, ds_test, cfg, test_traj_path)
        if cfg.DELTA_SOURCE == "train":
            if args.train_sampled_jsonl:
                print("[run] skip collecting train trajectories (use train_sampled_jsonl)")
            else:
                if ds_train is None:
                    raise RuntimeError("Train split is required when delta_source=train.")
                print("[run] phase1 collect(train)")
                collect_trajectories(model, tokenizer, ds_train, cfg, train_traj_path)

    # Phase 2
    if not args.skip_delta:
        print("[run] phase2 compute delta")
        if args.train_sampled_jsonl:
            trajectories = _build_trajectories_from_sampled_jsonl(
                args.train_sampled_jsonl,
                tokenizer,
                min_trajs_per_group=cfg.MIN_TRAJS_PER_GROUP,
                n_questions=args.n_questions,
            )
        else:
            traj_path = train_traj_path if cfg.DELTA_SOURCE == "train" else test_traj_path
            trajectories = _load_json(traj_path)
        compute_delta(model, tokenizer, trajectories, cfg, delta_path)

    # Phase 3
    if not args.skip_intervene:
        print("[run] phase3 intervention")
        if args.test_sampled_jsonl:
            test_trajectories = _build_trajectories_from_sampled_jsonl(
                args.test_sampled_jsonl,
                tokenizer,
                min_trajs_per_group=cfg.MIN_TRAJS_PER_GROUP,
                n_questions=args.n_questions,
            )
        else:
            test_trajectories = _load_json(test_traj_path)
        delta_dict = torch.load(delta_path, map_location=model_input_device(model))
        run_intervention(model, tokenizer, test_trajectories, delta_dict, cfg, inter_path)

    # Phase 4
    print("[run] phase4 analysis")
    analyze_and_plot(inter_path, cfg, save_dir)
    print("[run] done.")


def _build_runtime_config(args):
    data = {k: getattr(default_config, k) for k in dir(default_config) if k.isupper()}
    if args.model is not None:
        data["MODEL_NAME"] = args.model
    if args.delta_source is not None:
        data["DELTA_SOURCE"] = args.delta_source
    if args.device_map is not None:
        data["DEVICE_MAP"] = args.device_map
    return types.SimpleNamespace(**data)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _build_trajectories_from_sampled_jsonl(
    path: str, tokenizer, min_trajs_per_group: int, n_questions=None
):
    """
    Convert cal_metric-style jsonl records into Phase1-like trajectory structure.
    Input line format: {"prompt": str, "label": str, "predicts": list[str]}
    """
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if n_questions is not None:
        rows = rows[:n_questions]

    converted = []
    for qid, rec in enumerate(rows):
        prompt = rec.get("prompt", "")
        label = rec.get("label", "")
        predicts = rec.get("predicts", [])
        if isinstance(predicts, str):
            predicts = [predicts]
        if not isinstance(predicts, list):
            predicts = []

        system_prompt, user_prompt, ok = split_system_user_from_prompt(prompt)
        question = user_prompt if ok else prompt
        gold = parse_answer(label)
        correct_trajs, wrong_trajs = [], []

        for pred in predicts:
            if not isinstance(pred, str):
                continue
            pred_ans = parse_answer(pred)
            # Keep model input aligned with source file:
            # token sequence is built from original prompt + prediction text.
            full_text = f"{prompt}{pred}"
            token_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            item = {
                "text": pred,
                "full_text": full_text,
                "token_ids": token_ids,
                "predicted_answer": pred_ans,
            }
            if answers_equal(pred_ans, gold):
                correct_trajs.append(item)
            else:
                wrong_trajs.append(item)

        converted.append(
            {
                "question_id": qid,
                "question": question,
                "system_prompt": system_prompt if ok else None,
                "user_prompt": user_prompt if ok else None,
                "prompt": prompt,
                "ground_truth": gold,
                "correct_trajs": correct_trajs,
                "wrong_trajs": wrong_trajs,
                "skipped": (
                    len(correct_trajs) < min_trajs_per_group
                    or len(wrong_trajs) < min_trajs_per_group
                ),
            }
        )

    print(f"[run] converted {len(converted)} questions from sampled trajectories jsonl")
    return converted


def _load_local_qa_dataset(path: str, n_questions=None):
    p = Path(path)
    rows = []
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        obj = json.loads(p.read_text(encoding="utf-8"))
        rows = obj if isinstance(obj, list) else [obj]

    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        q = row.get("question", row.get("instruction"))
        a = row.get("answer", row.get("output"))
        if q is None or a is None:
            continue
        out.append({"question": str(q), "answer": str(a)})
    if n_questions is not None:
        out = out[:n_questions]
    print(f"[run] loaded {len(out)} local test questions")
    return out


if __name__ == "__main__":
    main()
