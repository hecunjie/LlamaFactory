import argparse
import os

import torch
import torch.distributed as dist

from dataset import DITPDataset, load_dataset_from_lf_info
from evaluate import evaluate_gsm8k
from model import load_model_and_tokenizer
from tokenizer_utils import register_pause_token
from trainer import train_model

try:
    import wandb
except ImportError:
    wandb = None


def build_samples_from_lf_rows(rows, tokenizer, template_style: str = "auto"):
    samples = []
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    use_manual_llama3 = False
    if template_style == "qa":
        use_chat_template = False
    elif template_style == "auto":
        use_chat_template = has_chat_template
    elif template_style in {"chat", "llama3"}:
        if template_style == "chat" and not has_chat_template:
            raise ValueError(
                "template_style=chat requires tokenizer.chat_template, "
                "but current tokenizer has no chat template."
            )
        use_chat_template = has_chat_template
        use_manual_llama3 = template_style == "llama3" and not has_chat_template
    else:
        raise ValueError(f"Unsupported template_style: {template_style}")

    eot_token = "<|eot_id|>" if tokenizer.convert_tokens_to_ids("<|eot_id|>") != tokenizer.unk_token_id else None

    for row in rows:
        if use_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        elif use_manual_llama3:
            prompt = (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{row['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            prompt = f"Q: {row['prompt']}\nA:"
        response = row["response"]
        if (use_chat_template or use_manual_llama3) and eot_token:
            if not response.endswith(eot_token):
                response = response + eot_token
        elif tokenizer.eos_token and not response.endswith(tokenizer.eos_token):
            response = response + tokenizer.eos_token
        samples.append({"prompt": prompt, "response": response})
    return samples


def main():
    parser = argparse.ArgumentParser(description="Train DIT / DIT-P on GSM8K-style data.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--mode", choices=["dit", "ditp"], default="ditp")
    parser.add_argument("--m_dit", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for cosine LR scheduler.",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument(
        "--dataset_info_path",
        type=str,
        default=os.path.join("..", "LlamaFactory", "data", "dataset_info.json"),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("..", "LlamaFactory", "data"),
    )
    parser.add_argument("--train_dataset_name", type=str, default="gsm8k_sft_train")
    parser.add_argument("--test_dataset_name", type=str, default="gsm8k_sft_test")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--template_style",
        choices=["auto", "chat", "qa", "llama3"],
        default="llama3",
        help="Prompt template style. auto=use tokenizer.chat_template when available.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=0,
        help="Save checkpoint every N optimizer steps. 0 disables step checkpointing.",
    )
    parser.add_argument(
        "--pause_selection",
        choices=["top_m", "prob_threshold"],
        default="top_m",
        help=(
            "Where to insert [PAUSE] in the target: "
            "top_m = paper/DIT: M positions with lowest log p(gold next token) (see --m_dit); "
            "prob_threshold = insert where p(gold next token) < --pause_prob_threshold, "
            "at most --m_dit inserts."
        ),
    )
    parser.add_argument(
        "--pause_prob_threshold",
        type=float,
        default=0.4,
        help="Used when --pause_selection prob_threshold (probability of gold next token).",
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--nextlat_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for NextLat-style latent alignment loss on [PAUSE] positions. "
            "0 disables this loss."
        ),
    )
    parser.add_argument(
        "--nextlat_loss_type",
        choices=["cosine", "mse"],
        default="cosine",
        help="Latent alignment loss type.",
    )
    parser.add_argument(
        "--nextlat_stopgrad_target",
        action="store_true",
        help="Stop gradient on target hidden states (next position).",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    if is_distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        is_main_process = dist.get_rank() == 0
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_main_process = True

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    pause_token_id, added = register_pause_token(tokenizer, model)
    if is_main_process:
        print(f"[PAUSE] token id: {pause_token_id}, newly_added: {added}")

    train_rows = load_dataset_from_lf_info(
        dataset_info_path=args.dataset_info_path,
        data_dir=args.data_dir,
        dataset_name=args.train_dataset_name,
    )
    test_rows = load_dataset_from_lf_info(
        dataset_info_path=args.dataset_info_path,
        data_dir=args.data_dir,
        dataset_name=args.test_dataset_name,
    )
    train_samples = build_samples_from_lf_rows(
        train_rows, tokenizer, template_style=args.template_style
    )
    test_samples = build_samples_from_lf_rows(
        test_rows, tokenizer, template_style=args.template_style
    )
    if is_main_process:
        print(f"Template style: {args.template_style}")
    wandb_run = None
    if is_main_process and args.wandb_project:
        if wandb is None:
            raise ImportError("wandb is not installed. Please `pip install wandb`.")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config={
                "model_name": args.model_name,
                "mode": args.mode,
                "m_dit": args.m_dit,
                "pause_selection": args.pause_selection,
                "pause_prob_threshold": args.pause_prob_threshold,
                "nextlat_weight": args.nextlat_weight,
                "nextlat_loss_type": args.nextlat_loss_type,
                "nextlat_stopgrad_target": args.nextlat_stopgrad_target,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "template_style": args.template_style,
            },
        )

    dataset = DITPDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        model=model,
        pause_token_id=pause_token_id,
        m_dit=args.m_dit,
        mode=args.mode,
        max_length=args.max_length,
        device=device,
        pause_selection=args.pause_selection,
        pause_prob_threshold=args.pause_prob_threshold,
    )
    if is_main_process:
        print(f"Prepared training samples: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError(
            "No valid training samples after preprocessing. "
            "Please check dataset paths, columns mapping in dataset_info.json, and max_length."
        )
    if is_main_process:
        print(
            "Pause stats | online_insertion=True, "
            f"mode={args.mode}, m_dit={args.m_dit}, pause_selection={args.pause_selection}"
            + (
                f", pause_prob_threshold={args.pause_prob_threshold}"
                if args.pause_selection == "prob_threshold"
                else ""
            )
        )

    model = train_model(
        model=model,
        dataset=dataset,
        epochs=args.epochs,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        batch_size=args.batch_size,
        device=device,
        save_steps=args.save_steps,
        save_path=args.save_path,
        tokenizer=tokenizer,
        pause_token_id=pause_token_id,
        mode=args.mode,
        m_dit=args.m_dit,
        pause_selection=args.pause_selection,
        pause_prob_threshold=args.pause_prob_threshold,
        nextlat_weight=args.nextlat_weight,
        nextlat_loss_type=args.nextlat_loss_type,
        nextlat_stopgrad_target=args.nextlat_stopgrad_target,
        wandb_run=wandb_run,
    )

    if is_main_process:
        model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
        print(f"Saved model to: {args.save_path}")

    if is_distributed and dist.is_initialized():
        dist.barrier()

    metrics = evaluate_gsm8k(
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        pause_token_id=pause_token_id,
        max_new_tokens=args.max_new_tokens,
        device=device,
        distributed=is_distributed,
    )
    if is_main_process:
        print(
            f"Eval | accuracy={metrics['accuracy']:.4f}, "
            f"avg_pause_count={metrics['avg_pause_count']:.4f}, "
            f"pause_nonzero_rate={metrics['pause_nonzero_rate']:.4f}, "
            f"n={metrics['num_samples']}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "eval/accuracy": metrics["accuracy"],
                    "eval/avg_pause_count": metrics["avg_pause_count"],
                    "eval/pause_nonzero_rate": metrics["pause_nonzero_rate"],
                    "eval/num_samples": metrics["num_samples"],
                }
            )

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if is_main_process and wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
