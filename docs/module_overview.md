# R$^{3}$ 代码模块说明

本文档概述当前仓库中主要代码目录与模块的功能定位、依赖关系及典型输入输出，方便后续完善实现与协同开发。

## 1. 数据处理与腐蚀模拟（`data_pipeline/`）

### 1.1 数据集读取（`data_pipeline/datasets/`）
- `base_dataset.py`：定义通用 `BasePMCDataset` 接口，负责索引构建、样本加载与 `collate`。
- `textvqa.py`, `chartqa.py`, `docvqa.py`：分别解析对应数据集的元数据（问题、答案、图像路径、OCR 等额外信息）。

### 1.2 模态腐蚀（`data_pipeline/corruption/`）
- `simulator.py`：`CorruptionSimulator` 根据配置概率触发模糊、遮挡、裁剪、OCR 噪声等操作，输出受损样本与不确定性报告。

### 1.3 伪文本生成（`data_pipeline/pseudo_text/`）
- `tokenizer.py`：将 OCR、密集描述等视觉信息语言化，生成带 `<OCR>`、`<CAP>` 标签的伪文本片段，用于检索与解释。

### 1.4 数据加载模块（`data_pipeline/dataloaders/`）
- `pmc_datamodule.py`：封装原始数据集、腐蚀模拟器与伪文本生成器，返回同时包含 `full_sample`、`corrupted_sample`、`pseudo_text`、`corruption_report` 的 batch。

## 2. 检索与证据过滤（`retrieval/`）

### 2.1 语料构建（`retrieval/indexer/`）
- `build_corpus.py`：将伪文本化后的样本写入 `corpus.jsonl`，后续可接入嵌入器/向量索引。

### 2.2 查询执行（`retrieval/query/`）
- `pseudo_text_retriever.py`：`PseudoTextRetriever` 用问题+伪文本构造查询，根据 token 重叠和噪声阈值返回 top-k 候选证据。

### 2.3 一致性过滤（`retrieval/filters/`）
- `consistency_filter.py`：过滤带 `[CONTRADICT]` 标记或与当前伪文本完全无重叠的证据，压制噪声。

## 3. 模型主体（`model/`）

### 3.1 主干适配（`model/backbones/`）
- `qwen3_vl_adapter.py`：封装 Qwen3-VL 主干，注入三路径增强与门控输出，接口统一接受问题 token、视觉 token、检索证据与腐蚀报告。

### 3.2 选择性重建模块（`model/modules/`）
- `tri_path_enhancer.py`：实现文本前缀、跨模注意记忆、潜空间填充三条路径，并根据门控值调节贡献。
- `adaptive_gate.py`：依据腐蚀不确定性与检索质量生成 `text/memory/imputation` 三路系数。

### 3.3 一致性正则（`model/losses/`）
- `consistency_regularizer.py`：对比完整模态与受损模态输出的 logits，约束推理一致性，降低幻觉。

## 4. 训练与优化（`train/`）

- `trainer.py`：高层训练循环，协调整个数据→腐蚀→检索→模型→损失链路，记录日志。
- `optimizer.py`：按主干/适配器拆分参数组，方便 LoRA/PEFT 微调场景设置差异化学习率。

## 5. 评测与指标（`eval/`, `metrics/`）

- `eval/evaluate_pmc.py`：在部分腐蚀场景评估准确率与幻觉率。
- `eval/evaluate_full.py`：完整模态对照实验。
- `metrics/hallucination.py`：依据检索证据统计幻觉比例，可在评测脚本中复用。

## 6. 配置与脚本（`configs/`, `scripts/`）

- `configs/default.yaml`：集中式配置范例，含数据、腐蚀、检索、训练、评测超参。
- `scripts/build_index.py`：命令行工具，读取伪文本样本构建检索语料。
- `scripts/export_results.py`：聚合评测指标输出 CSV/JSON，生成表格/可视化输入。

## 7. 推荐工作流

1. **准备数据**：下载目标数据集并按 `configs/default.yaml` 自定义路径，运行 `PMCDatamodule` 验证载入。
2. **构建语料**：利用 `pseudo_text/tokenizer.py` 生成伪文本，调用 `scripts/build_index.py` 写入检索语料并后续扩展向量索引。
3. **组装模型**：实例化 Qwen3-VL 主干，挂载 `TriPathEnhancer` 与 `AdaptiveGateController`。
4. **训练**：使用 `train/trainer.py`，配置优化器与损失项，观察一致性正则与QA损失。
5. **评测**：运行 `eval/evaluate_pmc.py`、`evaluate_full.py`，并通过 `scripts/export_results.py` 汇总结果，结合 `metrics/hallucination.py` 分析幻觉率。

以上结构确保“模拟受损→伪文本检索→选择性重建→一致性约束→评测”链路闭环，实现 R$^{3}$ 框架的落地与扩展。
