#!/usr/bin/env python3
"""
Convert JSONL with prompt + response into LlamaFactory SFT training format.

Input JSONL format (one JSON object per line):
  {"prompt": "<|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n", "response": "..."}

The script parses the ChatML-style prompt to extract system and user content,
then outputs Alpaca-style examples for LlamaFactory:
  - instruction: user message (required)
  - output: assistant response, including <add_think> etc. (required)
  - system: system message (optional; omit if empty)

Usage:
  python scripts/convert_prompt_response_jsonl_to_sft.py --input data.jsonl --output data_sft.json
  python scripts/convert_prompt_response_jsonl_to_sft.py --input data.jsonl --output data_sft.jsonl --format jsonl
"""

import argparse
import json
import os
import re


# ChatML-style: <|im_start|>role\ncontent<|im_end|>
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def trim_response_after_last_im_end(response: str) -> str:
    """Remove the last <im_end>/<|im_end|> token and anything after it."""
    if not response:
        return response
    candidates = (IM_END, "<im_end>")
    last_pos = -1
    for tok in candidates:
        pos = response.rfind(tok)
        if pos > last_pos:
            last_pos = pos
    if last_pos == -1:
        return response
    return response[:last_pos].rstrip()


def parse_chatml_prompt(prompt: str) -> tuple[str, str]:
    """Extract (system_content, user_content) from a ChatML prompt.

    Prompt is assumed to end with <|im_start|>assistant\n (no assistant content before response).
    """
    system_content = ""
    user_content = ""

    if not prompt.strip():
        return system_content, user_content

    # Split by role blocks
    pattern = re.compile(
        re.escape(IM_START) + r"(\w+)\n(.*?)" + re.escape(IM_END),
        re.DOTALL,
    )
    for m in pattern.finditer(prompt):
        role = m.group(1).strip().lower()
        content = m.group(2).strip()
        if role == "system":
            system_content = content
        elif role == "user":
            user_content = content
        # assistant block is typically empty in prompt (only the prefix)

    return system_content, user_content


def convert_line(line: str) -> dict | None:
    """Convert one JSONL line to Alpaca-style example. Returns None if invalid."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    prompt = obj.get("prompt")
    response = obj.get("response")
    if prompt is None or response is None:
        return None
    prompt = str(prompt)
    response = str(response).strip()
    response = trim_response_after_last_im_end(response)
    system_content, user_content = parse_chatml_prompt(prompt)
    out = {
        "instruction": user_content,
        "output": response,
    }
    if system_content:
        out["system"] = system_content
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert prompt+response JSONL to LlamaFactory SFT (Alpaca) format."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input JSONL file (each line: {\"prompt\": \"...\", \"response\": \"...\"})",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output JSON or JSONL file",
    )
    parser.add_argument(
        "--format",
        choices=("json", "jsonl"),
        default="json",
        help="Output format: json (one JSON array) or jsonl (one JSON object per line). Default: json",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    examples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            ex = convert_line(line)
            if ex is not None:
                examples.append(ex)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    if args.format == "jsonl":
        with open(args.output, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(examples)} examples -> {args.output}")
    if examples:
        sample = examples[0]
        print("  Sample keys:", list(sample.keys()))
        print("  instruction (first 80 chars):", (sample["instruction"][:80] + "..." if len(sample["instruction"]) > 80 else sample["instruction"]))
        print("  output (first 80 chars):", (sample["output"][:80] + "..." if len(sample["output"]) > 80 else sample["output"]))


if __name__ == "__main__":
    main()
