# R³: Retrieval–Reconstruction–Reasoning for Partial Modality Corruption

R³ 是一个围绕“部分模态损坏（PMC）”设计的多模态问答框架，目标是在真实场景下应对模态缺失、噪声与不一致所带来的幻觉问题。该实现遵循论文引言中的五大组件，并围绕 **Qwen3-VL 家族（默认使用 Qwen/Qwen3-VL-8B-Instruct，亦可切换到 Qwen2.5-VL 系列）** 主干进行了适配与微调（LoRA/QLoRA），同时具备可扩展的训练脚本与配置。

---

## 🧱 核心组件

| 模块 | 文件 | 核心职责 |
|------|------|-----------|
| **CorruptionSimulator** | [`r3/corruption_simulator.py`](r3/corruption_simulator.py) | 基于 token 特征预测置信度并按需注入噪声/遮挡，输出 `(features, confidence masks)`，在推理阶段仅估计置信度。 |
| **PseudoText Retrieval Module** | [`r3/retrieval_module.py`](r3/retrieval_module.py) | 以伪文本 + 置信度构建查询，噪声感知地筛选 top-K 证据，供后续前缀/记忆/填补使用。 |
| **SelectiveReconstructor** | [`r3/reconstructor.py`](r3/reconstructor.py) | 实现三路径融合：可解释前缀（Prefix Path）、跨模态记忆对齐（Memory Path）、语义填补（Latent Imputation Path），并通过门控函数调节。 |
| **R³ Model** | [`r3/r3_model.py`](r3/r3_model.py) | 将重建后的输入送入 Qwen3-VL/Qwen2.5-VL 主干，注入 LoRA/QLoRA，计算多目标损失 `L_total = L_task + λ_con L_cons + λ_aln L_align + λ_ref L_refuse + λ_ops L_ops`。 |
| **训练脚本** | [`train_r3.py`](train_r3.py) | 负责数据加载、腐蚀模拟、检索、重建、模型前向与训练循环，支持混合精度、课程学习和多 GPU。 |
| **评估脚本** | [`evaluate_r3.py`](evaluate_r3.py) | PMC场景下的模型评估，支持Accuracy/ANLS指标计算。 |
| **伪文本构建** | [`build_pseudo_text.py`](build_pseudo_text.py) | 离线构建伪文本语料库，支持OCR和Caption生成。 |

此外，原有的数据流水线（`data_pipeline/`）、伪文本生成器、Vision Encoder 依旧可复用，用于快速导入 TextVQA、ChartQA、DocVQA 等数据集。

---

## 🔄 端到端流程

flowchart LR
    subgraph Stage0["数据与配置"]
        D0[TextVQA/ChartQA/DocVQA] --> D1[BasePMCDataset]
        D1 --> D2[样本: {image, question, answer, extra}]
    end

    subgraph Stage1["腐蚀模拟 (CorruptionSimulator)"]
        D2 --> C1[视觉腐蚀: blur/noise/occlusion/crop]
        D2 --> C2[文本腐蚀: OCR noise/token drop/bbox drop]
        C1 --> C3[Ĩ]
        C2 --> C4[Q̃]
        C3 & C4 --> C5[不确定性掩码 U]
    end

    subgraph Stage2["伪文本与检索 (PseudoText Retrieval)"]
        C4 --> R1[PseudoTextBuilder]
        R1 --> R2[结构化条目 E={span,bbox,conf,src}]
        R2 --> R3[BM25 + Vector + Cross-Encoder]
        C5 --> R3
        R3 --> R4[top-k 证据 {E1..Ek}, 得分 s_i]
    end

    subgraph Stage3["选择性重建 (SelectiveReconstructor)"]
        R4 --> S1[Prefix Path: 证据前缀]
        R4 --> S2[Memory Path: 证据记忆库]
        C5 --> S3[Imputation Path: [IMPUTE_v/t]]
        S1 & S2 & S3 --> S4[增强后的多模输入]
    end

    subgraph Stage4["ReasoningHead (Qwen3-VL/Qwen2.5-VL + LoRA)"]
        S4 --> M1[Qwen-VL]
        M1 --> M2[Answer + Evidence IDs or REFUSE]
        M1 --> M3[多任务损失]
    end

---

## ⚙️ 训练配置与损失

训练包含 clean/corrupted 双分支，并使用统一答案或拒答标签。`train_r3.py` 中默认开启以下损失：

- `L_task`：主任务 SeqCE（Qwen 自带 loss）
- `L_consistency`：clean/corrupted logits 的 KL
- `L_align`：cross-attn（记忆得分）与检索得分对齐
- `L_refuse`：证据不足时的拒答监督（BCE）
- `L_ops`：图表/表格任务的算子一致性（CE/L1）

课程学习阶段：
1. Phase-0（clean）
2. Phase-1（轻/中度 PMC）
3. Phase-2（重度 PMC + 拒答监督）  
默认比例 3:2:1，可按 `configs/default.yaml` 调整。

优化器：AdamW (lr=2e-4, weight_decay=0.05) + Cosine decay + 5% warmup。  
训练建议：batch=2-4（受显存限制）、epochs=1-3、bf16 + gradient checkpoint。

---

## 📂 关键代码概览

r3/
├── corruption_simulator.py   # 视觉/文本腐蚀 + 不确定性掩码
├── retrieval_module.py       # PseudoText Retrieval + Noise-Aware Filter
├── reconstructor.py          # Prefix/Memory/Imputation 三路径融合
└── r3_model.py               # Qwen-VL + LoRA + 多任务损失

train_r3.py                   # 训练入口（含腐蚀、检索、重建）
evaluate_r3.py                # 评估脚本
build_pseudo_text.py          # 伪文本语料构建

其他重要模块：

- `data_pipeline/`：数据集解析（`datasets/`）、伪文本处理（`pseudo_text.py`）、视觉编码（`vision_encoder.py`）。
- `configs/`：配置文件，包含数据集、模型、训练等各模块参数。

---

## 🚀 使用指南

### 1. 安装依赖
pip install -r requirements.txt

### 2. 准备数据集
vim configs/default.yaml  # 设置 dataset.root, split, batch_size 等

### 3. 训练 R³
python train_r3.py --config configs/default.yaml --device cuda --output_dir checkpoints/r3_lora

### 4. 评估模型
python evaluate_r3.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/r3_lora \
  --split val \
  --predictions artifacts/val_preds.jsonl

### 5. 构建伪文本语料（可选）
python build_pseudo_text.py \
  --dataset_root ./data_pipeline/data/textvqa \
  --split train \
  --output ./artifacts/pseudo_text_train.jsonl \
  --enable_ocr \
  --caption_model qwen/Qwen2-VL-2B-Instruct

---

## ⚙️ 配置文件说明

基于 [`configs/default.yaml`](./configs/default.yaml)：

| 模块 | 参数 | 默认值 | 说明 |
|-----|------|--------|------|
| **数据集** | `root` | ./data_pipeline/data/textvqa | 数据集根目录 |
| | `split` | train | 训练数据分割 |
| | `batch_size` | 2 | 批处理大小 |
| | `num_workers` | 0 | 数据加载进程数 |
| **模型配置** | `name` | Qwen/Qwen3-VL-8B-Instruct | 主干网络名称 |
| | `lora_rank` | 32 | LoRA秩 |
| | `lora_alpha` | 16 | LoRA缩放因子 |
| | `hidden_size` | 4096 | 隐层维度 |
| | `provider` | huggingface / modelscope | 模型下载来源（HF 或魔搭） |
| | `token` | null | 私有仓库下载 Token（HF 或魔搭） |
| | `cache_dir` | ./hf_cache | 模型缓存目录 |
| | `revision` | null | 指定分支/版本 |
| | `local_files_only` | false | 仅使用本地缓存 |
| | `enable_corruption` | true | R³ Stage-1 开关 |
| | `enable_retrieval` | true | R³ Stage-2 开关 |
| | `enable_prefix` | true | 三路径：文本前缀 |
| | `enable_memory` | true | 三路径：证据记忆 |
| | `enable_imputation` | true | 三路径：语义填补 |
| | `enable_consistency` | true | 全模态一致性约束 |
| | `lambda_consistency` | 0.3 | 一致性损失系数 |
| | `top_k` | 3 | 检索证据数量 |
| **视觉编码** | `encoder` | openai/clip-vit-large-patch14 | 视觉编码器模型 |
| | `device` | cpu | 编码器运行设备 |
| | `cache_size` | 256 | 缓存大小 |
| **训练配置** | `epochs` | 1 | 训练轮数 |
| | `learning_rate` | 0.0002 | 学习率 |
| | `weight_decay` | 0.05 | 权重衰减 |
| | `warmup_ratio` | 0.05 | 预热比例 |
| | `log_interval` | 10 | 日志输出间隔 |

---

## 🧠 与 Qwen-VL 的适配

- Qwen3-VL/Qwen2.5-VL 由 `r3/r3_model.py` 加载，自动注入 `[IMPUTE_V]`、`[IMPUTE_T]` 等特征 token。
- LoRA/QLoRA 通过 `peft` 注入 `q_proj/k_proj/v_proj/o_proj/vision_proj`。
- 重建后的文本前缀与记忆向量会以 `inputs_embeds` 方式拼接到 Qwen 输入；imputation tokens 用于填补语义空洞。
- ReasoningHead 输出 `(answer, evidence_ids)` 或 REFUSE，并同步提供多任务损失。

---

## 🔧 数据处理流水线

### 数据集支持
- **TextVQA**: [`data_pipeline/datasets/textvqa.py`](data_pipeline/datasets/textvqa.py)
- **ChartQA**: [`data_pipeline/datasets/chartqa.py`](data_pipeline/datasets/chartqa.py)  
- **DocVQA**: [`data_pipeline/datasets/docvqa.py`](data_pipeline/datasets/docvqa.py)

### 伪文本处理
- **构建器**: [`data_pipeline/pseudo_text.py`](data_pipeline/pseudo_text.py) - 可按需离线聚合 OCR / Caption / Table 信息
- **在线伪文本**: `train_r3.R3Dataset` 会自动把 OCR/Captions 转成伪文本列表交给检索模块，无需额外索引

### 视觉特征处理
- **编码器**: [`data_pipeline/vision_encoder.py`](data_pipeline/vision_encoder.py) - 支持CLIP等预训练视觉模型
- **回退机制**: 图像无法访问时使用确定性随机特征，保证训练稳定性

### 模块开关
- `model.enable_corruption / enable_retrieval / enable_prefix / enable_memory / enable_imputation / enable_consistency` 控制 R³ 三大模块及一致性分支，可在 `configs/default.yaml` 中逐项消融。
- `model.top_k` 控制检索证据数量；其余超参（LoRA rank/dim、hidden_size 等）也集中于 `model` 段，便于统一管理。

### 权重下载与国内镜像
1. **HuggingFace**：执行 `huggingface-cli login` 并下载 `Qwen/Qwen3-VL-8B-Instruct`；若使用国内镜像，可提前设置  
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   export HF_HOME=./hf_cache
   ```
2. **魔搭 ModelScope**（“魔塔”）：在 `configs/default.yaml` 中把 `model.provider` 改为 `modelscope`，并填入 `model.token`（若仓库受限）。运行时会自动通过 `modelscope.snapshot_download` 缓存到 `model.cache_dir`。  
3. 如已手动下载本地权重，可把 `model.cache_dir` 指到相应路径或将 `model.name` 直接设为本地目录，训练脚本会优先使用本地文件。

---

## 📈 预期收益

- 在 PMC 条件下显著降低幻觉（≥25%）。
- 在 TextVQA、ChartQA、DocVQA、InfographicVQA 等基准上提升鲁棒性。
- 输出具备可解释性：答案附带证据 ID，可用于可视化或审计。
- 支持拒答机制，在证据不足时避免强制回答。

---

## 🛠️ 后续扩展

- 支持更强的向量检索（FAISS、ColBERT-v2），或引入多模态 cross-encoder。
- 将 Memory Path 接入 Qwen 内部 cross-attention（`attn_processors`），进一步提升对齐质量。
- 引入真实图像编码器（如 EVA-02）替代当前的 CLIP 视觉 embedding。
- 补充课程学习调度（clean→mild→heavy）与分布式训练支持。
- 扩展到更多多模态基准和任务类型。

---

## 📁 项目结构

R³/
├── 🚀 train_r3.py                    # 训练入口脚本
├── 📊 evaluate_r3.py                 # 评估脚本
├── 🔨 build_pseudo_text.py           # 伪文本语料构建
├── 📋 requirements.txt               # 依赖包列表
├── ⚙️ configs/                       # 配置文件
│   └── default.yaml                 # 默认配置
├── 🧠 r3/                           # R³核心模块
│   ├── corruption_simulator.py     # 模态腐蚀模拟器
│   ├── retrieval_module.py         # 混合检索系统
│   ├── reconstructor.py            # 选择性重建器
│   └── r3_model.py                  # R³主模型
└── 📊 data_pipeline/                # 数据处理流水线
    ├── datasets/                    # 数据集加载器
    │   ├── base_dataset.py         # 基础数据接口
    │   ├── textvqa.py              # TextVQA数据集
    │   ├── chartqa.py              # ChartQA数据集
    │   └── docvqa.py               # DocVQA数据集
    ├── pseudo_text.py              # 伪文本处理
    ├── vision_encoder.py           # 视觉特征编码
    └── data/                       # 示例数据

---

欢迎根据业务场景继续扩展 R³，也期待社区反馈，共同推进“可解释、可恢复”的多模态智能。
