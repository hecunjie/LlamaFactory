import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Replace <add_think> with <PAUSE> in JSONL text fields."
    )
    parser.add_argument("--input", required=True, help="Input .jsonl path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .jsonl path (default: <input>.pause.jsonl)",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["output"],
        help="JSON fields to perform replacement on (default: output)",
    )
    parser.add_argument(
        "--src_token",
        default="<add_think>",
        help="Source token to replace (default: <add_think>)",
    )
    parser.add_argument(
        "--dst_token",
        default="<PAUSE>",
        help="Destination token (default: <PAUSE>)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input file directly",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    if in_path.suffix.lower() != ".jsonl":
        raise ValueError("Only .jsonl is supported.")

    if args.inplace and args.output is not None:
        raise ValueError("--inplace and --output cannot be used together.")

    if args.inplace:
        out_path = in_path
    elif args.output:
        out_path = Path(args.output)
    else:
        out_path = in_path.with_suffix(".pause.jsonl")

    lines_out = []
    line_count = 0
    replaced_records = 0
    replaced_occurrences = 0

    with in_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line_count += 1
            s = raw.strip()
            if not s:
                continue
            obj = json.loads(s)
            changed = False
            for field in args.fields:
                val = obj.get(field)
                if isinstance(val, str):
                    cnt = val.count(args.src_token)
                    if cnt > 0:
                        obj[field] = val.replace(args.src_token, args.dst_token)
                        replaced_occurrences += cnt
                        changed = True
            if changed:
                replaced_records += 1
            lines_out.append(json.dumps(obj, ensure_ascii=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in lines_out:
            f.write(row + "\n")

    print(f"[DONE] lines={line_count}")
    print(f"[DONE] replaced_records={replaced_records}")
    print(f"[DONE] replaced_occurrences={replaced_occurrences}")
    print(f"[DONE] output={out_path}")


if __name__ == "__main__":
    main()
