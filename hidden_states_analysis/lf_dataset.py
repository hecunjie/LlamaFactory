# Copyright 2025 — hidden_states_analysis helper: load samples via LlamaFactory dataset_info.json
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from llamafactory.data.parser import DatasetAttr, get_dataset_list


def load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    obj = json.loads(text)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        data = obj.get("data")
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    raise ValueError(f"Unsupported JSON structure: {path}")


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_local_table(path: Path) -> list[dict[str, Any]]:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        return load_jsonl_records(path)
    if suf == ".json":
        return load_json_records(path)
    raise ValueError(f"Unsupported file type: {path}")


def example_to_prompt_predict(example: dict[str, Any], attr: DatasetAttr) -> dict[str, Any]:
    """Match AlpacaDatasetConverter: user content = prompt (+ optional query); assistant = response."""
    pcol = attr.prompt or "instruction"
    qcol = attr.query
    rcol = attr.response or "output"
    parts: list[str] = []
    if pcol and example.get(pcol) is not None:
        parts.append(str(example[pcol]))
    if qcol and example.get(qcol):
        parts.append(str(example[qcol]))
    prompt_text = "\n".join(parts)
    response_text = str(example.get(rcol, "")) if rcol else ""
    row: dict[str, Any] = {"prompt": prompt_text, "predict": response_text}
    if attr.system and example.get(attr.system) is not None:
        row["system"] = str(example[attr.system])
    if attr.tools and example.get(attr.tools) is not None:
        row["tools"] = str(example[attr.tools])
    return row


def load_dataset_rows(dataset_dir: str, dataset_name: str) -> tuple[list[dict[str, Any]], DatasetAttr]:
    r"""Load rows for a dataset name registered in dataset_info.json (same as LlamaFactory ``--dataset``)."""
    attrs = get_dataset_list([dataset_name], dataset_dir)
    attr = attrs[0]

    if attr.load_from == "file":
        local_path = os.path.join(dataset_dir, attr.dataset_name)
        if not os.path.isfile(local_path) and not os.path.isdir(local_path):
            raise FileNotFoundError(f"Dataset file not found: {local_path}")
        if os.path.isdir(local_path):
            raise ValueError(f"Directory dataset not supported in this script: {local_path}")
        raw = load_local_table(Path(local_path))
    elif attr.load_from == "hf_hub":
        from datasets import load_dataset

        ds = load_dataset(
            path=attr.dataset_name,
            name=attr.subset,
            data_dir=attr.folder,
            split=attr.split,
        )
        raw = ds.to_list()
    elif attr.load_from == "ms_hub":
        from modelscope import MsDataset  # type: ignore

        cache_dir = None
        ms_ds = MsDataset.load(
            dataset_name=attr.dataset_name,
            subset_name=attr.subset,
            data_dir=attr.folder,
            split=attr.split,
            cache_dir=cache_dir,
        )
        raw = ms_ds.to_hf_dataset().to_list()
    else:
        raise NotImplementedError(
            f"load_from={attr.load_from} not supported in hidden_states_analysis. "
            "Use a local file entry in dataset_info.json or hf_hub_url."
        )

    if attr.num_samples is not None:
        raw = raw[: int(attr.num_samples)]

    rows = [example_to_prompt_predict(ex, attr) for ex in raw]
    return rows, attr
