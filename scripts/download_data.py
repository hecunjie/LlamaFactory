# -*- coding: utf-8 -*-
"""
下载四个数学推理数据集到指定路径，并统一为 GSM 风格格式：
- instruction: 问题
- output: 推理过程 + "The answer is <答案>"

运行前：pip install datasets
"""

import os
import re
from datasets import Dataset, DatasetDict, load_dataset

# ====================== 配置部分 ======================
SAVE_ROOT = "./math_datasets"
# 如果想保存到其他地方，直接改这里

DATASETS = {
    "GSM8K-NL": "whynlp/gsm8k-aug-nl",       # 自然语言增强版，较大 (~387k)
    "ASDiv": "EleutherAI/asdiv",             # ASDiv-Aug 暂无确切HF repo，用原始版替代
    "MultiArith": "ChilleD/MultiArith",
    "SVAMP": "ChilleD/SVAMP"
}
# =======================================================


def _pick_first(example, keys):
    """从多个候选 key 中取第一个非空字段。"""
    for key in keys:
        value = example.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _build_gsm_style_output(answer_raw: str, rationale: str = "") -> str:
    """
    按 GSM 风格构造 output:
    - 若 answer_raw 含 ####，拆出最终答案；
    - 否则把 answer_raw 视作最终答案（或补充在 rationale 后）。
    """
    answer_raw = (answer_raw or "").strip()
    rationale = (rationale or "").strip()

    # GSM8K 常见格式: "... reasoning ... #### 42"
    match = re.search(r"####\s*(.+)$", answer_raw)
    if match:
        final_answer = match.group(1).strip()
        reasoning_part = answer_raw[:match.start()].strip()
        if reasoning_part:
            return f"{reasoning_part}\nThe answer is {final_answer}"
        return f"The answer is {final_answer}"

    # 若 answer_raw 本身已包含完整解释，优先使用它
    if answer_raw and not rationale:
        return answer_raw

    if rationale and answer_raw:
        return f"{rationale}\nThe answer is {answer_raw}"
    if answer_raw:
        return f"The answer is {answer_raw}"
    return rationale


def normalize_example(example, dataset_name: str):
    """
    统一转换为:
        {"instruction": "...", "output": "..."}
    """
    instruction = _pick_first(
        example,
        ["question", "Question", "problem", "Problem", "Body"],
    )
    # 某些数据集 Body + Question 分开存，优先拼接提升可读性
    body = _pick_first(example, ["Body", "body"])
    question = _pick_first(example, ["Question", "question"])
    if body and question:
        instruction = f"{body}\n{question}"

    # 可能的推理字段
    rationale = _pick_first(
        example,
        ["rationale", "solution", "Solution", "explanation", "Equation", "formula", "Formula"],
    )
    # 可能的答案字段
    answer_raw = _pick_first(
        example,
        ["answer", "Answer", "final_ans", "final_answer", "label", "target"],
    )
    output = _build_gsm_style_output(answer_raw=answer_raw, rationale=rationale)

    return {"instruction": instruction, "output": output}


def convert_dataset_to_gsm_style(ds, dataset_name: str):
    """把 Dataset / DatasetDict 转换为 GSM 风格字段。"""
    if isinstance(ds, DatasetDict):
        converted = {}
        for split, split_ds in ds.items():
            converted[split] = split_ds.map(
                lambda ex: normalize_example(ex, dataset_name),
                remove_columns=split_ds.column_names,
            )
        return DatasetDict(converted)

    if isinstance(ds, Dataset):
        return ds.map(
            lambda ex: normalize_example(ex, dataset_name),
            remove_columns=ds.column_names,
        )

    return ds

def main():
    # 创建根目录
    os.makedirs(SAVE_ROOT, exist_ok=True)
    print(f"所有数据集将保存到: {os.path.abspath(SAVE_ROOT)}\n")

    for name, repo_id in DATASETS.items():
        print(f"\n=== 下载并保存 {name} ({repo_id}) ===")
        try:
            # 尝试加载数据集（默认加载所有 split）
            dataset = load_dataset(repo_id, trust_remote_code=True)

            # 统一格式为 GSM 风格
            dataset = convert_dataset_to_gsm_style(dataset, name)

            save_path = os.path.join(SAVE_ROOT, name.replace("-", "_"))

            if isinstance(dataset, DatasetDict):  # 有 train/test/validation 等 split
                dataset.save_to_disk(save_path)
                print(f"已保存 (多 split, GSM 格式): {save_path}")
            else:  # 单个 dataset
                dataset.save_to_disk(save_path)
                print(f"已保存 (GSM 格式): {save_path}")

            # 打印一些基本信息
            print(f"  splits: {list(dataset.keys() if isinstance(dataset, DatasetDict) else ['single'])}")
            if isinstance(dataset, DatasetDict) and "train" in dataset:
                print(f"  train 示例数: {len(dataset['train'])}")
                if len(dataset["train"]) > 0:
                    print(f"  train 样例字段: {dataset['train'].column_names}")
            if isinstance(dataset, DatasetDict) and "test" in dataset:
                print(f"  test 示例数: {len(dataset['test'])}")
            if isinstance(dataset, Dataset) and len(dataset) > 0:
                print(f"  单一 split 样例字段: {dataset.column_names}")
                
        except Exception as e:
            print(f"下载 {name} 失败: {e}")
            print("可能原因：网络问题、无此数据集、需要登录HF（可运行 huggingface-cli login）")
            continue

    print("\n下载完成！")
    print(f"文件夹结构示例：")
    print(f"{SAVE_ROOT}/")
    for name in DATASETS:
        print(f"  ├─ {name.replace('-', '_')}/   ← Arrow 格式，可用 load_from_disk 读取")

if __name__ == "__main__":
    main()