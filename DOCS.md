## R³ End-to-End Pipeline (Detailed)

This document is meant to mirror the implementation so you can line it up with the paper intro. Every stage below maps to concrete files/functions.

### 1) Dataset, Config, and Offline Pseudo-Text
- Configure `configs/default.yaml`:
  - `dataset.root/split/batch_size`
  - `dataset.pseudo_corpus` (optional): path to JSONL produced by `build_pseudo_text.py` (one object per line with `doc_id`, `pseudo_text`)
  - `model.*`: backbone (e.g., `Qwen/Qwen3-VL-8B-Instruct`), LoRA, `vision_tokens`, `hidden_size`, `top_k`, consistency weight, etc.
- Base loaders are in `data_pipeline/datasets/` (e.g., `textvqa.py`).

### 2) Sample Construction (`R3Dataset` in `train_r3.py`)
For each sample `{id, image_path, question, answer, extra}`:
1. **Pseudo-text assembly (priority order)**:
   - If `dataset.pseudo_corpus` is set and contains this `doc_id`, use that pseudo-text.
   - Else pull inline OCR/captions/context from `extra`.
   - Else fallback via `PseudoTextBuilder` (`r3/retrieval_module.py`): uses question/id anchors if nothing else.
2. **Pre-encoding corruption (data-level, `data_pipeline/corruptions.py`)**:
   - Image: `ImageCorruptor` applies random occlusion rectangle + optional Gaussian blur → produce clean image and corrupted image copies.
   - Pseudo-text: `PseudoTextCorruptor` drops a fraction of entries for the corrupted branch; the question string is untouched.
3. **Branch packaging**:
   - `clean`: full pseudo-text + clean image (PIL), labels, meta (`vision_tokens`, `hidden_size`).
   - `corrupted`: dropped pseudo-text + occluded/blurred image, same labels/meta.
   - Both branches keep `question` unchanged.

### 3) Collation (`collate_fn` in `train_r3.py`)
- Batches contain two parallel dicts (`clean`, `corrupted`) with lists for `question`, `labels`, `pseudo_text`, `images` (PIL), `vision_tokens`, `hidden_size`.

### 4) Vision Encoding (before model)
- In `R3Trainer.compute_loss`, `_get_vision_embeddings` calls `model.base_vlm.encode_images` on the provided PIL images (clean vs corrupted) using the Qwen vision tower. No external encoder is used; tokens are resized to `vision_tokens` and matched to `hidden_size`.

### 5) Model Structure (`r3/r3_model.py`)
- **Base**: `BaseVLM` wraps Qwen-VL + LoRA (`base_vlm.py`).
- **Simulator**: `UncertaintyAwareCorruptionSimulator` now only predicts token-wise confidence maps for vision/text (no token corruption).
- **Retrieval**: `PseudoTextRetrievalModule` hashes pseudo-text strings via the LM embedding layer; noise-aware scoring boosts evidence when `mask_intensity = 1 - img_conf` is high.
- **Reconstruction**: `SelectiveReconstruction` fuses prefix/memory/imputation with adaptive gates, producing `inputs_embeds` + `attention_mask` for the backbone.
- **Dual-branch forward**:
  - Clean branch (`is_clean_branch=True`, `no_grad`): simulator bypass for corruption (conf only), retrieval skipped, reconstruction skipped; returns teacher hidden.
  - Corrupted branch (`is_clean_branch=False`): simulator (conf only) → retrieval → reconstruction → backbone forward → logits/hidden.

### 6) Training Loop (`R3Trainer.compute_loss` in `train_r3.py`)
1. Tokenize prompts (question + answer) separately for clean/corrupted splits.
2. Encode images (clean vs corrupted) via `base_vlm.encode_images` → vision tokens.
3. Forward passes:
   - Teacher (clean, no_grad): `pooled_hidden_clean`.
   - Student (corrupted): `loss_task`, `pooled_hidden_corrupt`, retrieval artifacts.
4. Loss: `loss = loss_task + lambda_c * MSE(pooled_hidden_clean.detach(), pooled_hidden_corrupt)`.
5. Curriculum (`CurriculumScheduler`): epoch 0–1 → `lambda_c=0`; epoch ≥2 → `lambda_c=1` (optionally adjust dropout configs).
6. Runner: HuggingFace `TrainingArguments` + `R3Trainer` + `collate_fn`; call `trainer.train()`.

### 7) Retrieval & Offline Corpus Alignment
- If `dataset.pseudo_corpus` is provided, `R3Dataset` loads pseudo-text by `doc_id` from the JSONL (output of `build_pseudo_text.py`), so training-time retrieval consumes the same offline corpus.

### 8) Inference / Evaluation Options
- To simulate missing modality across backbones, keep `corruptions.py` enabled on inputs; the in-model simulator will only emit confidence maps.
- To run clean, disable/skip `corruptions.py` and feed clean images/pseudo-text; the clean branch still provides teacher features if desired.

### 9) Key Files (where to look)
- `train_r3.py` — dataset wrapper, collate, trainer, curriculum, corpus loading.
- `data_pipeline/corruptions.py` — pre-encoding occlusion/blur + pseudo-text drop.
- `r3/r3_model.py` — dual-branch forward (clean teacher / corrupted student).
- `r3/retrieval_module.py` — pseudo-text builder + noise-aware retrieval.
- `r3/reconstructor.py` — prefix/memory/imputation fusion.
- `r3/corruption_simulator.py` — confidence estimation (no token corruption).
