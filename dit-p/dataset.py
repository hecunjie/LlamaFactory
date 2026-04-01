import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset


def load_dataset_from_lf_info(
    dataset_info_path: str,
    data_dir: str,
    dataset_name: str,
) -> List[Dict[str, str]]:
    """
    Load dataset by LlamaFactory's dataset_info.json schema.
    Expected fields:
      - file_name
      - columns.prompt / columns.response (optional, defaults to prompt/response)
    """
    info = json.loads(Path(dataset_info_path).read_text(encoding="utf-8"))
    if dataset_name not in info:
        raise ValueError(f"dataset_name '{dataset_name}' not found in {dataset_info_path}")

    meta = info[dataset_name]
    file_name = meta.get("file_name")
    if not file_name:
        raise ValueError(f"dataset '{dataset_name}' missing file_name in dataset_info.json")

    columns = meta.get("columns", {})
    prompt_col = columns.get("prompt", "prompt")
    response_col = columns.get("response", "response")
    file_path = Path(data_dir) / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"dataset file not found: {file_path}")

    suffix = file_path.suffix.lower()
    rows: List[dict] = []
    if suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(f"Expected list JSON in {file_path}")
    elif suffix == ".jsonl":
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    samples: List[Dict[str, str]] = []
    for row in rows:
        if prompt_col in row and response_col in row:
            samples.append(
                {
                    "prompt": str(row[prompt_col]),
                    "response": str(row[response_col]),
                }
            )
    return samples


class DITPDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Dict[str, str]],
        tokenizer,
        model,
        pause_token_id: int,
        m_dit: int = 5,
        mode: str = "ditp",
        max_length: int = 1024,
        device: str = "cuda",
    ) -> None:
        self.features = []
        self.pad_token_id = tokenizer.pad_token_id
        self.mode = mode
        self.pause_token_id = pause_token_id
        self.total_pause_inserted = 0
        model.eval()
        model = model.to(device)

        for sample in samples:
            prompt = sample["prompt"]
            response = sample["response"]

            prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(response, add_special_tokens=False)
            full_ids = prefix_ids + target_ids
            if len(full_ids) > max_length:
                overflow = len(full_ids) - max_length
                if overflow >= len(target_ids):
                    continue
                target_ids = target_ids[overflow:]
                full_ids = prefix_ids + target_ids

            if not target_ids:
                continue

            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[:, :-1, :]
                shifted_labels = input_ids[:, 1:]
                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_likelihood = log_probs.gather(
                    dim=-1, index=shifted_labels.unsqueeze(-1)
                ).squeeze(-1)

            prefix_len = len(prefix_ids)
            target_ll = token_log_likelihood[0, max(prefix_len - 1, 0) :].tolist()
            if not target_ll:
                selected = set()
            else:
                pairs = sorted(enumerate(target_ll), key=lambda x: x[1])
                selected = {idx for idx, _ in pairs[: min(m_dit, len(pairs))]}

            new_input_ids = []
            new_labels = []
            for i, tid in enumerate(prefix_ids):
                new_input_ids.append(tid)
                new_labels.append(-100)

            for i, tid in enumerate(target_ids):
                if i in selected:
                    new_input_ids.append(pause_token_id)
                    new_labels.append(pause_token_id if mode == "ditp" else -100)
                    self.total_pause_inserted += 1
                new_input_ids.append(tid)
                new_labels.append(tid)

            self.features.append(
                {
                    "input_ids": new_input_ids,
                    "labels": new_labels,
                    "attention_mask": [1] * len(new_input_ids),
                }
            )

        self.total_tokens = sum(len(x["input_ids"]) for x in self.features)
        self.pause_density = (
            self.total_pause_inserted / max(self.total_tokens, 1) if self.features else 0.0
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx]

    def collate_fn(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        labels = []
        attention_masks = []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
            attention_masks.append(item["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }
