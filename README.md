# R$^{3}$：鲁棒多模态检索-重建-推理框架

R$^{3}$（Retrieval, Reconstruction, Reasoning）面向**部分模态腐蚀（Partial Modality Corruption, PMC）**场景，通过“腐蚀感知 + 伪文本检索 + 选择性重建”提升多模态问答的稳健性与可解释性。

## 项目亮点
- **腐蚀模拟与不确定性估计**：`data_pipeline/corruption/simulator.py` 提供结构化模糊、遮挡、OCR 噪声等操作，生成受损样本与模态置信度。
- **伪文本检索链路**：`data_pipeline/pseudo_text/tokenizer.py` 将视觉内容语言化，`retrieval/query/pseudo_text_retriever.py` 结合噪声感知过滤完善外部证据。
- **三路径重建**：`model/modules/tri_path_enhancer.py` 与 `adaptive_gate.py` 在 Qwen3-VL 适配器中实现文本前缀、跨模注意记忆、潜空间填充三条增益路径，并以一致性正则抑制幻觉。
- **端到端训练/评测脚手架**：`train/trainer.py` 串联腐蚀→检索→模型→损失，`eval/` 与 `metrics/` 提供 PMC/全模态基准评估与幻觉指标。

更多模块细节见 `docs/module_overview.md`。

## 快速开始
1. **安装依赖**：在 AutoDL 或本地执行 `pip install -r requirements.txt`，提供最小运行环境（PyTorch + PyYAML）。
2. **准备数据**：按照 `configs/default.yaml` 设置数据路径，确保 `dataset.root` 指向 TextVQA/ChartQA/DocVQA 的解压目录，可通过 `configs/default.yaml` 中的 `dataset` 字段自定义 batch 大小与 worker 数。
3. **构建检索语料**：生成伪文本后执行 `python scripts/build_index.py --input samples.jsonl --output runs/index` 得到 `corpus.jsonl`，作为 `scripts/train.py --index` 的输入。
4. **启动训练**：使用统一脚本 `python scripts/train.py --config configs/default.yaml --index runs/index/corpus.jsonl --device cuda`，脚本会解析配置、初始化 `PMCDatamodule`、伪文本检索器与 `Trainer` 并自动开始迭代，可通过 `--epochs/--batch-size/--num-workers` 快速覆盖。
5. **评测与导出结果**：使用 `eval/evaluate_pmc.py`、`eval/evaluate_full.py` 获取指标，借助 `scripts/export_results.py` 聚合输出。

本仓库当前提供完整的端到端脚手架（含特征构建、训练脚本与依赖清单），可在此基础上替换真实模型与算子以复现论文实验。
