<div align="center">

# R³: Retrieval · Reconstruction · Reasoning

_Robust multimodal VQA under partial-modality corruption (PMC) with Qwen3-VL._

</div>

---

## What’s New (Current Codebase)
- **Backbone-native vision**: images are encoded directly by Qwen-VL’s own vision tower (`BaseVLM.encode_images`), no external CLIP encoder required; precomputed embeddings are still accepted when available.
- **Dual-branch teacher–student**: `r3/r3_model.py` exposes `forward(..., is_clean_branch)`, where the clean branch runs in `torch.no_grad()` (teacher) and bypasses simulator/retrieval, and the corrupted branch runs corruption → retrieval → reconstruction.
- **Noise-aware retrieval**: retrieval down-weights low-confidence text, boosts regions with low visual confidence via `mask_intensity = 1 - img_conf`, and selects top‑k pseudo-text evidence.
- **Pseudo-text auto-fill**: `PseudoTextBuilder` now lives in `r3/retrieval_module.py`; when datasets lack OCR/captions, pseudo-text is synthesized from question/id and optional on-the-fly caption/OCR hooks.
- **Curriculum-ready trainer**: `R3Trainer.compute_loss` performs a single-step forward/backward with task CE + consistency MSE; `CurriculumScheduler` toggles corruption/dropout and consistency weight by epoch.

---

## Architecture at a Glance

**Stage 0 – Inputs**
- Images + questions (+ optional labels).
- Optional: OCR tokens, captions, page context; otherwise pseudo-text is auto-built或从离线伪文本库加载（`build_pseudo_text.py` 输出 JSONL）。

**Stage 1 – Corruption & Uncertainty**
- 数据层缺模态（`data_pipeline/corruptions.py`）: 原图遮挡/模糊，伪文本随机丢弃；问题文本保持不变。
- 模型内: `UncertaintyAwareCorruptionSimulator` 仅输出 vision/text 置信度，不再改写 token。

**Stage 2 – Noise-Aware Retrieval**
- `PseudoTextRetrievalModule` encodes pseudo-text entries with the LM embedding layer.
- 支持两种源：批内伪文本或离线伪文本库（`model.retrieval_corpus_path` 载入 `build_pseudo_text.py` 的 JSONL），可选 FAISS。
- Query = 置信度加权文本特征；在 `mask_intensity = 1 - img_conf` 高的区域提升分数；Top‑k 证据返回。

**Stage 3 – Selective Reconstruction**
- `SelectiveReconstruction` merges three paths: prefix (encoded evidence), memory (cross-attn to evidence), and imputation tokens gated by confidence.
- Output `inputs_embeds` + `attention_mask`; 后接 `TriPathReasoner`（2 层 8 头 Transformer）进一步细化。

**Stage 4 – Backbone Reasoning**
- Qwen3-VL/Qwen2.5-VL with LoRA/QLoRA (flash-attn 2 compatible).
- Clean branch: direct embeddings, no simulator/retrieval, teacher features for consistency.
- Corrupted branch: full R³ stack, logits and pooled hidden for loss.

**Loss (per step)**
```python
feat_clean  = model(..., is_clean_branch=True)["pooled_hidden"]   # no_grad teacher
student_out = model(..., is_clean_branch=False)
loss_task   = student_out["loss"]
loss_cons   = mse_loss(feat_clean.detach(), student_out["pooled_hidden"])
loss = loss_task + lambda_c * loss_cons
```
Curriculum: epochs 0–1 use low dropout + no consistency; epoch ≥2 uses higher dropout + λ=1.

---

## Repository Map

- `r3/base_vlm.py` — Qwen-VL 封装，LoRA 注入（含 FFN/vision_proj），原生视觉塔编码。
- `r3/corruption_simulator.py` — 置信度预测（不改写 token）。
- `r3/retrieval_module.py` — `PseudoTextBuilder` + 噪声感知检索，支持离线伪文本库/FAISS。
- `r3/reconstructor.py` — Prefix / Memory / Imputation 融合 + `TriPathReasoner`。
- `r3/r3_model.py` — 双分支前向（clean 教师 / corrupted 学生）。
- `train_r3.py` — 数据封装、collate、`R3Trainer`、课程调度、离线伪文本加载。
- `build_pseudo_text.py` — 离线伪文本生成（支持 textvqa/mp_docvqa/infovqa，OCR+caption 可选）。
- `data_pipeline/` — 数据集适配器、伪文本工具、`corruptions.py`（前置缺模态模拟）。
- `DOCS.md` — 详细流水线解读。

---

## Data & Pseudo-Text

### Page-as-Evidence & Auto-Fill
- `PseudoTextBuilder` fields: OCR tokens, captions, `context_evidence` (e.g., neighbor pages), plus fallbacks to `[Q] question` / `[ID] doc_id` when no annotations exist.
- For MP-DocVQA, add neighbor page OCR to `extra.context_evidence` during preprocessing.

### One-Click Pseudo-Text (optional)
```bash
python build_pseudo_text.py \
  --dataset_root ./data_pipeline/data/textvqa \
  --split train \
  --output ./artifacts/pseudo_text_train.jsonl \
  --enable_ocr \
  --caption_model Qwen/Qwen2-VL-2B-Instruct
```
This will inject OCR (pytesseract) and captions when missing, then save JSONL pseudo-text for fast loading.
At training time you can point `dataset.pseudo_corpus` in `configs/default.yaml` to this JSONL to ensure offline pseudo-text is used for retrieval.

---

## Training Quickstart

1) **Install deps**
```bash
pip install -r requirements.txt
```

2) **Configure** (`configs/default.yaml`)
- `dataset.root`, `split`, `batch_size`
- `model.name` (e.g., `Qwen/Qwen3-VL-8B-Instruct`)
- `model.vision_tokens`, `model.hidden_size`, `model.top_k`
- No separate `vision` section is needed; the backbone vision tower is always used.
- 若有离线伪文本库：设置 `dataset.pseudo_corpus` 与 `model.retrieval_corpus_path` 指向 `build_pseudo_text.py` 生成的 JSONL；可选 `model.retrieval_cache_path` 启用 FAISS。

3) **Run training**
```bash
python train_r3.py --config configs/default.yaml --device cuda --output_dir checkpoints/r3_lora
```
`train_r3.py` will:
- 离线载入 `dataset.pseudo_corpus`（若配置），否则在线构建伪文本。
- 使用基座视觉塔编码，无外部编码器。
- 同步 clean teacher + corrupted student，一步前后向，课程调度控制 λ/难度。

4) **Evaluate**
```bash
python evaluate_r3.py --config configs/default.yaml --checkpoint checkpoints/r3_lora --split val
```

---

## Inference Flow (PMC-Aware)
```text
image → Qwen vision tower → vision tokens
question + pseudo-text → embeddings
conf maps ← simulator(vision, text)         # no corruption at inference
retrieval(pseudo, conf) → top-k evidence
reconstruction(prefix/memory/impute) → inputs_embeds
TriPathReasoner → refined embeds
backbone generation → answer (+ evidence scores)
```

---

## Dataflow (Training Step)
```text
1) Load sample → build/auto-fill pseudo-text (OCR/caption/context or fallback to [Q]/[ID]); 若设置 `dataset.pseudo_corpus` 则直接加载 JSONL；可选前置缺模态：图像遮挡/模糊 + 伪文本丢弃（`data_pipeline/corruptions.py`）
2) Tokenize question+answer → input_ids, attention_mask, labels
3) Encode images with Qwen vision tower → vision_tokens (resize/match hidden_size)
4) Split branches:
   - Clean branch (teacher, no_grad):
       * Merge text + vision → backbone forward → pooled_hidden_clean
   - Corrupted branch (student):
       * Simulator: vision/text embeddings → conf maps（不改写 token）
       * Retrieval: 离线/在线伪文本，噪声感知 scoring → top-k 证据
       * Reconstruction: prefix/memory/imputation gating → fused inputs_embeds
       * TriPathReasoner: refine fused tokens
       * Backbone forward → logits, pooled_hidden_corrupt, loss_task
5) Loss:
   loss_consistency = MSE(pooled_hidden_clean.detach(), pooled_hidden_corrupt)
   total_loss = loss_task + lambda_c * loss_consistency
```

---

## Key Flags & Tips
- **Vision**: `model.vision_tokens` 需与下游上下文匹配，维度匹配 `model.hidden_size`。
- **Retrieval**: `model.top_k` 控制证据数；`model.retrieval_corpus_path` 指向离线伪文本库；`model.retrieval_cache_path` 可启用 FAISS；自动构建/离线库均保证非空检索。
- **Corruption**: 数据层缺模态（遮挡/模糊 + 伪文本丢弃）可通过 `corruptions.py`；模型内仅置信度估计，课程调度控制 λ/遮挡强度。
- **Consistency**: `model.config.lambda_consistency` 由 CurriculumScheduler 控制（热身 0，之后 1）。
- **LoRA**: 注入 `q_proj/k_proj/v_proj/o_proj/vision_proj/gate_proj/up_proj/down_proj`；按需调整 `lora_rank/alpha`。
- **Config hygiene**: 不需要单独 vision 配置块；使用 `model.*` 管理视觉 token 数/隐藏维度。

---

## Troubleshooting
- **No vision embeddings?** Ensure `model.name` is a vision-capable Qwen variant and weights are downloaded; `encode_images` will raise if the vision tower is absent.
- **Empty pseudo-text?** Builder will fallback to `[Q] question`/`[ID] doc_id`, but for best results run `build_pseudo_text.py` with OCR/caption.
- **OOM**: reduce `vision_tokens`, sequence length, or LoRA rank; keep teacher branch in `no_grad` (default).

---

## Roadmap
- Page-level curriculum for MP-DocVQA (dynamic neighbor context span).
- Vector-store backed retrieval (FAISS/ColBERT) with confidence-aware reranking.
- Distributed training + checkpoint save/load for adapters and reconstruction modules.
