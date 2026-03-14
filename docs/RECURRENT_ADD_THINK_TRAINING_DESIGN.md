# Recurrent Training with `<add_think>` — 设计说明（已实现）

## 目标

在训练时，当序列中出现 `<add_think>` 时：

1. **该位置的输入**：不用 `<add_think>` 的 embedding，而是用**上一个 token 的 hidden state** 作为输入。
2. **下一个 token 的输入**：不用“下一个 token”的 embedding，而是用**当前（即 `<add_think>` 位置）的 hidden state** 作为输入。

即：用 hidden 替代两处 embedding——`<add_think>` 处和紧跟其后的那个 token 处。

## 前向语义（按位置）

设序列为 `... t_{k-1}, <add_think>, t_{k+1}, t_{k+2}, ...`（位置 k 是 `<add_think>`，k+1 是下一个真实 token）。

| 位置 | 输入 | 输出 |
|------|------|------|
| ... | embed(t) | hidden |
| k-1 | embed(t_{k-1}) | hidden_{k-1} |
| k   | **hidden_{k-1}**（非 embed(<add_think>)） | hidden_k |
| k+1 | **hidden_k**（非 embed(t_{k+1})） | hidden_{k+1} |
| k+2 | embed(t_{k+2}) | ... |

因此：

- 每个 `<add_think>` 会带来两步“hidden 作为输入”：一步在 `<add_think>`，一步在下一个 token。
- 前向必须**按步进行**（或按“被 `<add_think>` 分隔的段”做多段 forward），无法对整句做一次标准 causal forward，**效率会明显变慢**。

## 损失

- **忽略 `<add_think>` 上的损失**：若预测目标是 `<add_think>`，该位置不参与 loss（labels 中对应位置为 `IGNORE_INDEX`）。
- **其余位置**：按常规 next-token 预测算交叉熵。
  - 特别地，**“下一个 token”位置**（即用 hidden_k 作为输入、预测 t_{k+1} 的那一步）**要算 loss**，用 t_{k+1} 作为 target，从而用 CE 指导“给定 hidden 时预测下一个 token”的能力。

即：只忽略“目标为 `<add_think>`”的 CE；其他位置（包括用 hidden 预测真实 token 的位置）都参与训练。

## 实现要点（拟在 `training_step` 中做的）

1. **开关**  
   - 在 `FinetuningArguments` 中增加一项，例如 `recurrent_add_think_training: bool = False`。  
   - 在 `training_step` 里增加一个 `elif` 分支：当启用该开关且当前 batch 的 `input_ids` 中包含 `<add_think>` 时，走“recurrent add_think”分支，否则保持现有 SFT / latent_chain 逻辑。

2. **前向（recurrent 分支）**  
   - 获取 `<add_think>` 的 token id，并在序列中找出所有 `<add_think>` 的位置。  
   - 按样本、按位置逐步 forward：  
     - 若当前位是 `<add_think>`：输入 = 上一步的 last hidden state（reshape 成 1,1,dim）；  
     - 若上一步是 `<add_think>` 的 hidden 输出：当前步输入 = 该 hidden（即“下一个 token 的输入也是当前位置的 hidden_states”）；  
     - 否则：输入 = 当前 token 的 embedding。  
   - 使用 KV cache 串起各步，避免重复计算。  
   - 每步得到 logits，用于后续 loss；需要 next-token 的 logits 与 labels 对齐（见下）。

3. **Loss 计算（recurrent 分支）**  
   - 与标准 SFT 一致：`logits[i]` 预测的是 position `i+1` 的 token。  
   - `labels`：  
     - 若 position `i+1` 是 `<add_think>`，则 `labels[i] = IGNORE_INDEX`；  
     - 否则 `labels[i] = input_ids[i+1]`（或当前数据里已有的 next-token label）。  
   - 只对 `labels != IGNORE_INDEX` 的位置做 CE，对 `logits` 做 shift 后与 `labels` 对齐（与现有 `compute_loss` 的 shift 方式一致）。

4. **与现有 latent_chain 的关系**  
   - 现有 `_forward_with_latent_chain` 用的是另一套 special token 和 `special_token_mask`，且有一层 `latent_hidden_norm`。  
   - Recurrent add_think 训练是**独立分支**：由“是否启用 recurrent_add_think_training + 是否含 `<add_think>`”触发，不共用 latent_chain 的 Phase 1/2/3，也不要求 `latent_hidden_norm`（除非你后续希望在这里也加一个 norm）。  
   - 数据格式：需要 `input_ids` / `labels` 中已经插入了 `<add_think>`（例如由 `mark_low_confidence_positions` 生成），且 labels 里 `<add_think>` 对应位置为 `IGNORE_INDEX`。

5. **效率与实现方式**  
   - 若一个序列中有多个 `<add_think>`，每个都会引入 2 步“hidden 输入”，整段前向本质是逐 token 或按段串行，无法整句一次矩阵乘，因此会比普通 SFT 慢很多。  
   - 实现时可考虑：  
     - 按样本循环，每个样本内部按位置步进（或按“段”步进，段内用一次 forward，段间用 hidden 接上），并做好 padding / batch 处理或接受 per-sample 串行；  
     - 或先做 per-sample 实现，batch 内逐样本执行 recurrent 前向并聚合 loss，再 backward。

## 需要你确认的点

1. **“下一个 token 的输入也是当前位置的 hidden_states”**  
   是否就是：紧跟在 `<add_think>` 后面的那个 token 位置，**不**用该 token 的 embedding，而用 `<add_think>` 位置算出的 hidden 作为输入？  
   - 若是，上面表格和 loss 设计就按此实现。

2. **多个连续 `<add_think>`**  
   若出现 `... <add_think> <add_think> t ...`，是否规则为：  
   - 第一个 `<add_think>`：输入 = hidden_{prev}，输出 = h1；  
   - 第二个 `<add_think>`：输入 = h1，输出 = h2；  
   - t：输入 = h2（即仍用“上一个位置”的 hidden）？  
   若是，则逻辑可以统一为：“凡当前 token 是 `<add_think>` 或上一位置是 hidden 输入，则当前输入 = 上一位置的 hidden”。

3. **是否需要对 hidden 做归一化/投影**  
   现有 latent_chain 有 `latent_hidden_norm`。Recurrent add_think 是否也加一层可学习的 norm（或直接用 raw hidden）？若不加，就按“直接用上一 token 的 last-layer hidden”实现。

4. **Batch 内混合**  
   同一 batch 里部分样本含 `<add_think>`、部分不含时，是否：  
   - 含的走 recurrent 分支（可能 per-sample 循环），不含的走标准 forward，最后 loss 取平均？  
   还是要求数据/配置上保证“recurrent 模式下 batch 内全部是含 `<add_think>` 的样本”？

## 实现摘要

- **开关**: `FinetuningArguments.recurrent_add_think_training`；为 True 时整 batch 走 recurrent 分支。
- **可学习 LayerNorm**: 首次进入该分支时在 unwrapped 上注册 `add_think_hidden_norm`（`nn.LayerNorm(hidden_size)`），作为子模块参与训练。
- **前向**: `_forward_recurrent_add_think` 按段 + 两步（`<add_think>` 与下一 token）做 forward，收集每位置 logits，再对 `labels != IGNORE_INDEX` 做 CE。
- **损失**: 与标准 SFT 一致，`logits[i]` 预测下一 token；labels 中 `<add_think>` 目标位置为 `IGNORE_INDEX`，其余位置参与 CE。
