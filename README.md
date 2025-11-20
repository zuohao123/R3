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
- Optional: OCR tokens, captions, page context; otherwise pseudo-text is auto-built.

**Stage 1 – Corruption & Uncertainty**
- Pre-encoding: raw images → optional occlusion/blur; pseudo-text → dropout (data_pipeline/corruptions.py).
- In-model: `UncertaintyAwareCorruptionSimulator` now only predicts token-wise confidence (no further corruption on tokens).

**Stage 2 – Noise-Aware Retrieval**
- `PseudoTextRetrievalModule` encodes pseudo-text entries with the LM embedding layer.
- Query = confidence-weighted text features; scores boosted where `mask_intensity` (1 − img_conf) is high; select top‑k evidence.

**Stage 3 – Selective Reconstruction**
- `SelectiveReconstruction` merges three paths: prefix (encoded evidence), memory (cross-attn to evidence), and imputation tokens gated by confidence.
- Output `inputs_embeds` + `attention_mask` ready for the backbone.

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

- `r3/base_vlm.py` — Qwen-VL wrapper, LoRA injection, **vision tower encoding** (`encode_images`), processor handling.
- `r3/corruption_simulator.py` — confidence heads + corruption injection.
- `r3/retrieval_module.py` — `PseudoTextBuilder` (auto pseudo-text) + noise-aware retrieval.
- `r3/reconstructor.py` — prefix / memory / imputation fusion with adaptive gates.
- `r3/r3_model.py` — dual-branch forward (clean teacher vs corrupted student).
- `train_r3.py` — dataset wrapper, collate, `R3Trainer`, curriculum scheduler.
- `build_pseudo_text.py` — offline pseudo-text generation (OCR + caption optional).
- `data_pipeline/` — dataset loaders, pseudo-text utilities, and `corruptions.py` for pre-encoding modality drops.

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

3) **Run training**
```bash
python train_r3.py --config configs/default.yaml --device cuda --output_dir checkpoints/r3_lora
```
`train_r3.py` will:
- Build pseudo-text on the fly (if missing OCR/captions).
- Use the backbone vision tower to encode images (no external encoder needed).
- Run clean teacher + corrupted student in one step with curriculum toggling.

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
backbone generation → answer (+ evidence scores)
```

---

## Dataflow (Training Step)
```text
1) Load sample → build/auto-fill pseudo-text (OCR/caption/context or fallback to [Q]/[ID]); apply pre-encoding corruptions (image occlusion/blur + pseudo-text dropout) via `data_pipeline/corruptions.py`
2) Tokenize question+answer → input_ids, attention_mask, labels
3) Encode images with Qwen vision tower → vision_tokens (resize/match hidden_size)
4) Split branches:
   - Clean branch (teacher, no_grad):
       * Merge text + vision → backbone forward → pooled_hidden_clean
   - Corrupted branch (student):
       * Simulator: vision/text embeddings → conf maps + feature dropout/noise
       * Pseudo-text drop at data level (partial missing evidence)
       * Retrieval: noise-aware scoring → top-k evidence embeddings
       * Reconstruction: prefix/memory/imputation gating → fused inputs_embeds
       * Backbone forward → logits, pooled_hidden_corrupt, loss_task
5) Loss:
   loss_consistency = MSE(pooled_hidden_clean.detach(), pooled_hidden_corrupt)
   total_loss = loss_task + lambda_c * loss_consistency
```

---

## Key Flags & Tips
- **Vision**: set `model.vision_tokens` to match downstream context; hidden_dim must match `model.hidden_size`.
- **Retrieval**: `model.top_k` controls evidence count; pseudo-text auto-build ensures non-empty retrieval.
- **Corruption**: `simulator.config.image_dropout/text_dropout` tuned by `CurriculumScheduler`.
- **Consistency**: `model.config.lambda_consistency` updated by curriculum (0 during warmup, 1 later).
- **LoRA**: injected into `q_proj/k_proj/v_proj/o_proj/vision_proj`; adjust `lora_rank/alpha` in config.
- **Config hygiene**: any legacy `vision` block in configs is ignored; only `model.*` controls vision token count/size.

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
