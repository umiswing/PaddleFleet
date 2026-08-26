# SLASH MLA — Model & Training Scaling

纯 MLA + 全 Full attention 基线三规模（0.6B / 1.7B / 10B-A1B），用于研究 Attention 结构与长文。
本文档记录 **scaling 方法论**，便于后续加规模 / 调超参 / 换阶段。

## 1. 模型结构 (model_config.json)

| | 0.6B dense | 1.7B dense | 10B-A1B MoE |
|---|---|---|---|
| dir | `model_configs/mla_0p6B_dense/` | `model_configs/mla_1p7B_dense/` | `model_configs/mla_10B_A1B/` |
| hidden / layers | 1024 / 28 | 2048 / 28 | 2048 / 24 |
| FFN (dense) | 3072 | 6144 | 5120 (仅第0层) |
| MoE | — | — | 256 experts, top-8, 1 shared, `moe_latent_size=512` (LatentMoE), `first_k_dense_replace=1` |
| MLA (DeepSeek 标准) | qk_nope=128, qk_rope=64, v=128, kv_lora=512, q_lora=512, 16 heads | 同左 | 同左 |
| tie_word_embeddings | true | true | false |
| vocab | 102400 | 102400 | 102400 |
| 总参 / 激活 | ~0.58B / ~0.58B | ~1.61B / ~1.61B | 10.1B / 1.13B |

要点：
- `head_dim` 字段对 MLA 无效（代码 `multi_latent_attention.py` 用 `q_head_dim=qk_nope+qk_rope` 覆盖）。
- `num_key_value_heads` 对 MLA 无效（MLA=MHA，KV 从 latent 重建到全头，代码硬编码 kv=1）。
- dense 用 `first_k_dense_replace=L`（全 dense，**勿再设 moe_layer_freq**，否则 transformer_config.py:1389
  报 `Cannot specify both`）；MoE 用 `first_k_dense_replace=k` + `moe_layer_freq=<int>`。
- **LatentMoE (`moe_latent_size=512`) 是 10B sizing 的决定性因素**：专家在 512 维 latent 计算，
  把每专家参数砍半，总参从 ~19.6B 降到 10.1B。不要关掉它。

## 2. Scaling 方法论 — 采用 Kimi Linear (arxiv 2510.26692)

Kimi Linear 的做法：
- **架构对比走 Chinchilla scaling law** (Hoffmann 2022)：在 Moonlight 架构 MoE (激活 8/64,
  Muon 优化器, ctx 4096) 上训 5 个不同规模，每个用 grid search 调到最优，拟合
  **Loss = A·C^(−α)** (C = 算力, PFLOP·days)。结果 MLA `2.3092·C^-0.0536`,
  Kimi Linear `2.2879·C^-0.0527` → KDA 约 1.16× 算力效率。
- **生产模型不套公式**：单个 48B/3B-active MoE (8/256+1shared, 首层 dense, MuonClip, WSD),
  固定 lr=1.1e-3, global batch=32M tokens, 训 1.4T→5.7T tokens, ctx4096 后做长上下文激活阶段。
- **优化器 Muon/MuonClip + WSD 调度**（scaling 网格与生产都用 WSD）。

### Table 2 隐含的超参幂律（对其 5 点反向回归；论文只显式给 Loss-C 律）
以 **N = 激活参数（不含 embedding，单位 M）** 为自变量：

| 量 | 幂律 | 说明 |
|---|---|---|
| 数据 D (B tokens) | `D = 0.0135 · N^1.236` | token/激活参 ~60→75 (均值 ~71)，远高于经典 Chinchilla 的 20 |
| 学习率 lr | `lr = 26.4e-3 · N^(−0.398)` | 越大越小 |
| batch (序列, ctx4096) | `bsz = 4.59 · N^0.667` | 越大越大 |

Kimi Table 2 原始 5 点（复现校验用）：

| N_active(M) | Head=Layer | Hidden | Tokens | lr | batch(序列) |
|---|---|---|---|---|---|
| 653 | 16 | 1216 | 38.8B | 2.006e-3 | 336 |
| 878 | 18 | 1376 | 59.8B | 1.790e-3 | 432 |
| 1100 | 20 | 1536 | 85.2B | 1.617e-3 | 512 |
| 1400 | 22 | 1632 | 102.5B | 1.486e-3 | 576 |
| 1700 | 24 | 1776 | 128.0B | 1.371e-3 | 640 |

## 3. 套用到本项目三规模（8K base 阶段, 8机64卡）

基座数据 = **Ultra FineWeb-edu (m102k / vocab 102400, 非 BFD)**
（`SLASH/datas/vocsize-102400-ctx-8k.{zy,ratio}`，2700 parts，单源 ratio=1.0，来自 afs_ro
`000100-98-ernie5_tk_m102k_vid6p1-...-for_h5/F000000010000000005/`，~138B tokens）。
非 BFD（无 bfd_frag_start key，reader 走 concat）→ seqlen 自由，本阶段取 **8192**。
model_config `max_position_embeddings=8192`, `rope_theta=10000`, `vocab_size=102400`（与数据一致）。
注: 另有一份 m200k/201216 的同 ID FineWeb-edu（ctx8k BFD, `vocsize-201216-ctx-8k`），词表与本项目
（102400）不符，已弃用（orphan，可删）。

约束（框架公式）：`sharding_parallel_size × gradient_accumulation_steps × per_device_train_batch_size = global_batch_size`
- `sharding_parallel_size` = 总卡数 = **64** → `global_batch_size` 必须是 64 的倍数。
- MoE 档：`expert_model_parallel_size(8)` 整除 64；`n_routed_experts(256)` 整除 EP(8)。
- token-batch 沿用 Kimi 计划（seqlen 从 4096→8192，序列 gbs 相应减半以保持 tokens/step）。

| 规模 | N_active(不含embed) | lr | gbs(序列) | per_dev×acc | tok/step | max_steps | 实际 tokens |
|---|---|---|---|---|---|---|---|
| 0.6B dense | ~475M | 2.271e-3 | 128 | 2×1 | 1.05M | 26000 | 27B |
| 1.7B dense | ~1400M | 1.477e-3 | 256 | 2×2 | 2.10M | 49500 | 104B |
| 10B-A1B | ~710M | 1.936e-3 | 192 | 1×3 | 1.57M | 29000 | 46B |

公共：`lr_scheduler=wsd`, `warmup_steps=2000`, `min_lr=0`, `save_steps=2000`, `max_seq_length=8192`,
优化器 muon (`muon_qkv_update_mode=split_head`)。

显存兜底（per_device×acc 恒等于表中值即可换）：
- 1.7B 若放不下 9×4096 → `per_device=3, acc=3`。
- 10B 显存宽裕可 `per_device=6, acc=1`（默认取保守 3×2）。

## 4. 后续 scaling 时怎么用本文档

1. **加新规模**：算出该模型的激活参数 N（M, 不含 embedding），代入
   `lr=26.4e-3·N^-0.398`、`bsz(序列)=4.59·N^0.667`、`D(B)=0.0135·N^1.236`。
2. **batch 落地**：把 `bsz` 向上/就近取 64 的倍数，拆成 `per_device×acc`（64×acc×per_device=gbs）。
3. **max_steps** = `D×1e9 / (gbs×seqlen)`。
4. **换训练阶段（长文）**：改 `max_seq_length` 与 model_config 的 `max_position_embeddings`；
   长文阶段需把 `rope_theta` 提到 1e6 起或改 YARN（当前 4K 阶段 rope_theta=10000 是有意为之）。
   注意 seqlen 必须 == 数据的 BFD context_length（换 seqlen 要换对应长度的数据表）。
5. **MoE 数据口径**：用激活参而非总参（Chinchilla, 同 Kimi Table 2）。若走"过训"生产模型，
   token 量远超 compute-optimal（Kimi 生产 1.4T/5.7T），需自行加大 max_steps。

## 5. 长文多阶段 (8K base → 32K → 128K)

从 **8K** base checkpoint 续训做长上下文激活。**原则：base→32K 可断（重开一个受控 cosine），
32K→128K 必须连续（同一条 cosine，不重启 lr）。**

> 说明：Kimi Linear 本身用 **NoPE**（MLA 无位置编码，位置交给 KDA），刻意不调 rope/YaRN；
> 本项目是 **RoPE-MLA**，故长文走标准 RoPE recipe（升 theta / YaRN），非照搬 Kimi。

### 阶梯表（每段各 20B tokens；tokens/step 均 ~2.1M）

| 项 | 32K 段 (前 20B) | 128K 段 (后 20B) |
|---|---|---|
| `max_seq_length` / model `max_position_embeddings` | 32768 | 131072 |
| rope | `rope_type=rope`, **`rope_theta=64000`**(从8K外推4x) | **`rope_type=yarn`, `rope_theta=500000`**, `rotary_scaling_factor=4`, `original_max_position_embeddings=32768` |
| `global_batch_size`(序列) | 64 | 16 |
| 并行 (64卡) | CP=1 → `64×acc×per_device=64` | **CP=8 → DP=8**, `8×acc×per_device=16` (per_device=1,acc=2) |
| max_steps | ~9500 | ~9500 |
| 数据 | 32K 表(待定) | 128K 表(待定) |

### LR：整段一条连续 cosine（跨 32K+128K）
- `lr_scheduler: cosine`；warmup ~200（只在长文段最开始做一次）。
- peak ≈ 8K base peak 的 ~1/10（10B: ~1.9e-4 / 1.7B: ~1.5e-4 / 0.6B: ~2.3e-4）。
- **cosine 总步数 = 32K+128K 合计（40B, ~19000 步）**，min_lr ≈ 1e-5。
- 边界(20B处)自然衰到 `(peak+min)/2 ≈ 1.05e-4`，128K **续接该值继续下衰到 1e-5，不重启**。
- 实现：128K 单独 launch 时，cosine 总步数设成 40B 的总步数、并**恢复 checkpoint 里的
  `LR_Scheduler` 状态**（pretraining_trainer 支持），使 step 接着走、lr 连续。
- YaRN 的 `original_max_position_embeddings=32768`（= 32K 阶段长度，从 32K 外推），factor=131072/32768=4。

### 落地要点
- rope 参数在 **model_config.json**（`rope_theta`/`rope_type`/`max_position_embeddings`，YaRN 还需
  `rotary_scaling_factor`/`original_max_position_embeddings`）：每段复制一份改过的 model_config 目录，
  或用 `--kwargs` 覆盖。
- **128K 必须开 context parallel**（单卡放不下 131072，10B 尤甚）：`context_parallel_size=8`，
  此时 batch 公式变 `DP(=64/CP) × acc × per_device`。CP 语义以框架为准。
- **数据**：仓库现只有 4k/8k 表，32K/128K 的 BFD 表需另造（seqlen 必须 == BFD context_length）。

## 6. 开源/闭源 解耦约定 + 启动方式

**边界**：
- **开源** `third_party/PaddleFleet/`（引擎 + attention/模型算法 `src/paddlefleet/transformer/`）、
  `third_party/PaddleFormers/`（`GPTModelProvider` 基类）；本研究目录 `Research/SLASH/` 也在开源侧。
- **闭源** `exp2p20_pull/` 外层：`ernie5/`（pretrain.py/trainer/数据 reader/`utils/config.py`）、
  `fleet_model/ernie5_v2/modeling.py`（ERNIE↔Fleet 配置桥 `Ernie5V2Provider`）、`script/`、`conf/`、tokenizer。

**约定：SLASH 只"新增 + 调用"，绝不改闭源。** 研究改动落点：
1. 新 attention/模型算法 → `PaddleFleet/src/paddlefleet/transformer/`（开源）。
2. 实验配置/数据/yaml → `Research/SLASH/{model_configs,conf,datas}`（开源）。
3. 启动 → 只读调用闭源 `script/train.sh`（见下 run.sh）。

**唯一耦合点 = 闭源配置桥** `fleet_model/ernie5_v2/modeling.py`：
- 新 model_config 字段若**复用 PaddleFleet/PaddleFormers 已有同名字段**，通用引擎透传，**不用改闭源**。
- 只有字段需要**改名 / 特殊 value 处理**（走 `transform_rules`/`apply_ernie_config_overrides`）才被迫动闭源——尽量避免。
- 数据侧：坚持标准 H5 + filelist/weight 契约，就不用改闭源 reader。

**启动器** `Research/SLASH/run.sh`（开源；自动 cd 到 exp2p20_pull 根，只读调 `script/train.sh --configs`）：
```bash
bash third_party/PaddleFleet/Research/SLASH/run.sh <scale> <stage> [额外 train.sh 参数]
#   scale: 0p6B | 1p7B | 10B     stage: 8k | 32k | 128k
bash .../run.sh 10B 8k
bash .../run.sh 10B 32k  --kwargs resume_from_checkpoint=<8K ckpt>
bash .../run.sh 10B 128k --kwargs resume_from_checkpoint=<32K ckpt>   # 续 32K, cosine 连续
```
32k/128k 会自动叠加对应 base（`--configs base stage`）。软耦合：运行 cwd 是 exp2p20_pull，
故 yaml 内 `data_filelist`/`model_name_or_path` 是相对该闭源根（当前用相对路径，够用）。

## 注意事项 / 未决
- 幂律指数是对 Kimi Table 2 的反向回归；论文只显式给 Loss-C 律。数值已核对能复现 Table 2。
- Kimi 用 **MuonClip**，本项目用 muon（无 clip）；10B 那档若早期出现 loss 尖刺，考虑降 lr 或加 clip。
- **纯 dense 的 ernie5_v2 在仓库内无现成配置**（全 MoE）；首次跑 0.6B/1.7B 前建议先 build 验证
  dense+MLA 构图正常。
- 数据表用 4K：`conf/allinone/eb5/EB5Pretrain-260624-a1b-v18p3-4k`。

参考：Kimi Linear: An Expressive, Efficient Attention Architecture, arXiv:2510.26692.
