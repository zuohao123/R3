# R$^{3}$: Robust Multimodal Retrieval-Reconstruction-Reasoning Framework

R$^{3}$ (Retrieval, Reconstruction, Reasoning) is designed for **Partial Modality Corruption (PMC)** scenarios, enhancing the robustness and interpretability of multimodal question answering through "corruption-aware + pseudo-text retrieval + selective reconstruction".

## Project Highlights
- **Corruption Simulation & Uncertainty Estimation**: `data_pipeline/corruption/simulator.py` provides structured blur, occlusion, OCR noise operations to generate corrupted samples with modality confidence scores.
- **Pseudo-text Retrieval Pipeline**: `data_pipeline/pseudo_text/tokenizer.py` linguifies visual content, while `retrieval/query/pseudo_text_retriever.py` combines noise-aware filtering to enhance external evidence.
- **Tri-path Reconstruction**: `model/modules/tri_path_enhancer.py` and `adaptive_gate.py` implement three enhancement paths in Qwen3-VL adapters: text prefix, cross-modal attention memory, and latent space filling, with consistency regularization to suppress hallucinations.
- **End-to-end Training/Evaluation Scaffold**: `train/trainer.py` chains corruption→retrieval→model→loss, while `eval/` and `metrics/` provide PMC/full-modal benchmark evaluation and hallucination metrics.

For more module details, see `docs/module_overview.md`.

## Quick Start
1. **Install Dependencies**: Execute `pip install -r requirements.txt` on AutoDL or locally to provide minimal runtime environment (PyTorch + PyYAML).
2. **Prepare Data**: Set data paths according to `configs/default.yaml`, ensure `dataset.root` points to the extracted directory of TextVQA/ChartQA/DocVQA. Customize batch size and worker count through the `dataset` field in `configs/default.yaml`.
3. **Build Retrieval Corpus**: After generating pseudo-text, execute `python scripts/build_index.py --input samples.jsonl --output runs/index` to get `corpus.jsonl`, which serves as input for `scripts/train.py --index`.
4. **Start Training**: Use the unified script `python scripts/train.py --config configs/default.yaml --index runs/index/corpus.jsonl --device cuda`. The script will parse configuration, initialize `PMCDatamodule`, pseudo-text retriever and `Trainer`, then automatically start iterations. Quick overrides available via `--epochs/--batch-size/--num-workers`.
5. **Evaluation & Export Results**: Use `eval/evaluate_pmc.py`, `eval/evaluate_full.py` to obtain metrics, and leverage `scripts/export_results.py` to aggregate outputs.

This repository currently provides a complete end-to-end scaffold (including feature construction, training scripts, and dependency lists), which can be used as a foundation to replace with real models and operators to reproduce paper experiments.
