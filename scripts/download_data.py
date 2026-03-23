# -*- coding: utf-8 -*-
"""
下载四个数学推理数据集到指定路径
- GSM8K-NL: whynlp/gsm8k-aug-nl (自然语言增强版)
- ASDiv-Aug: EleutherAI/asdiv (原始ASDiv作为替代，Aug版暂无公开HF repo)
- MultiArith: ChilleD/MultiArith
- SVAMP: ChilleD/SVAMP

运行前：pip install datasets
"""

import os
from datasets import load_dataset, concatenate_datasets

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

def main():
    # 创建根目录
    os.makedirs(SAVE_ROOT, exist_ok=True)
    print(f"所有数据集将保存到: {os.path.abspath(SAVE_ROOT)}\n")

    for name, repo_id in DATASETS.items():
        print(f"\n=== 下载并保存 {name} ({repo_id}) ===")
        try:
            # 尝试加载数据集（默认加载所有 split）
            dataset = load_dataset(repo_id, trust_remote_code=True)
            
            # 如果有多个 split，合并成一个 dict 或只保存 train/test
            save_path = os.path.join(SAVE_ROOT, name.replace("-", "_"))
            
            if isinstance(dataset, dict):  # 有 train/test/validation 等 split
                dataset.save_to_disk(save_path)
                print(f"已保存 (多 split): {save_path}")
            else:  # 单个 dataset
                dataset.save_to_disk(save_path)
                print(f"已保存: {save_path}")
            
            # 打印一些基本信息
            print(f"  splits: {list(dataset.keys() if isinstance(dataset, dict) else ['single'])}")
            if "train" in dataset:
                print(f"  train 示例数: {len(dataset['train'])}")
            if "test" in dataset:
                print(f"  test 示例数: {len(dataset['test'])}")
                
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