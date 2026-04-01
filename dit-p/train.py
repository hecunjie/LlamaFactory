import argparse
import os

import torch

from dataset import DITPDataset, load_dataset_from_lf_info
from evaluate import evaluate_gsm8k
from model import load_model_and_tokenizer
from tokenizer_utils import register_pause_token
from trainer import train_model


def build_samples_from_lf_rows(rows, tokenizer):
    samples = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        response = row["response"]
        if tokenizer.eos_token and not response.endswith(tokenizer.eos_token):
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
        "--save_steps",
        type=int,
        default=0,
        help="Save checkpoint every N optimizer steps. 0 disables step checkpointing.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    pause_token_id, added = register_pause_token(tokenizer, model)
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
    train_samples = build_samples_from_lf_rows(train_rows, tokenizer)
    test_samples = build_samples_from_lf_rows(test_rows, tokenizer)

    dataset = DITPDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        model=model,
        pause_token_id=pause_token_id,
        m_dit=args.m_dit,
        mode=args.mode,
        max_length=args.max_length,
        device=device,
    )
    print(f"Prepared training samples: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError(
            "No valid training samples after preprocessing. "
            "Please check dataset paths, columns mapping in dataset_info.json, and max_length."
        )
    print(
        f"Pause stats | inserted={dataset.total_pause_inserted}, "
        f"density={dataset.pause_density:.6f}, mode={args.mode}, m_dit={args.m_dit}"
    )

    train_model(
        model=model,
        dataset=dataset,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        save_steps=args.save_steps,
        save_path=args.save_path,
        tokenizer=tokenizer,
    )

    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"Saved model to: {args.save_path}")

    metrics = evaluate_gsm8k(
        model=model,
        tokenizer=tokenizer,
        samples=test_samples,
        pause_token_id=pause_token_id,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    print(
        f"Eval | accuracy={metrics['accuracy']:.4f}, "
        f"avg_pause_count={metrics['avg_pause_count']:.4f}, "
        f"pause_nonzero_rate={metrics['pause_nonzero_rate']:.4f}, "
        f"n={metrics['num_samples']}"
    )


if __name__ == "__main__":
    main()
