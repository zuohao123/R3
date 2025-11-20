## R³ Pipeline Cheat Sheet

### 1) Dataset & Config
- Set `configs/default.yaml`: `dataset.root/split/batch_size`, optional `dataset.pseudo_corpus` (JSONL from `build_pseudo_text.py`), and `model.*` (backbone/LoRA/vision_tokens).
- Loaders live under `data_pipeline/datasets/` (e.g., `textvqa.py`).

### 2) Data Preparation (R3Dataset in `train_r3.py`)
1. Load sample `{image_path, question, answer, extra}`.
2. Build pseudo-text:
   - Prefer `dataset.pseudo_corpus` by `doc_id` (offline JSONL).
   - Else inline OCR/captions/context from `extra`.
   - Else `PseudoTextBuilder` fallback (`[Q]/[ID]`).
3. Apply **pre-encoding corruption** (`data_pipeline/corruptions.py`):
   - Image: occlusion + Gaussian blur (`ImageCorruptor`) → clean/corrupted copies.
   - Pseudo-text: dropout for corrupted branch (`PseudoTextCorruptor`); question intact.
4. Return two branches: `clean` (full pseudo-text, clean image) and `corrupted` (dropped pseudo-text, occluded/blurred image).

### 3) Collation (`collate_fn`)
- Batches carry lists for `question`, `labels`, `pseudo_text`, `images`, `vision_tokens`, `hidden_size`, split into clean/corrupted.

### 4) Vision Encoding (before model)
- In `R3Trainer.compute_loss`, `_get_vision_embeddings` calls `model.base_vlm.encode_images` on the PIL images (clean vs corrupted) using the Qwen vision tower. No external encoder is used.

### 5) Model Structure (`r3/r3_model.py`)
- **Base**: Qwen-VL + LoRA (`BaseVLM`).
- **Simulator**: `UncertaintyAwareCorruptionSimulator` now **only outputs confidence maps** (no token corruption).
- **Retrieval**: `PseudoTextRetrievalModule` consumes pseudo-text strings; noise-aware scoring uses `mask_intensity = 1 - img_conf` to boost low-visibility regions.
- **Reconstruction**: `SelectiveReconstruction` merges prefix/memory/imputation with adaptive gates.
- **Dual-branch forward**:
  - Clean (`is_clean_branch=True`, `no_grad`): bypass simulator/retrieval, provide teacher hidden.
  - Corrupted (`is_clean_branch=False`): simulator (confidence only) → retrieval → reconstruction → backbone.

### 6) Training Loop (`R3Trainer.compute_loss`)
1. Tokenize question+answer for clean/corrupted splits.
2. Encode images (clean vs corrupted) via `base_vlm.encode_images`.
3. Forward:
   - Teacher (clean, no_grad): `pooled_hidden_clean`.
   - Student (corrupted): `loss_task`, `pooled_hidden_corrupt`.
4. Loss: `loss = loss_task + lambda_c * MSE(pooled_hidden_clean.detach(), pooled_hidden_corrupt)`.
5. Curriculum (`CurriculumScheduler`): epoch 0–1 → `lambda_c=0`; epoch ≥2 → `lambda_c=1`; also toggles dropout configs if needed.
6. Runner: `TrainingArguments` + `R3Trainer` + `collate_fn`; call `trainer.train()`.

### 7) Retrieval & Offline Corpus
- If `dataset.pseudo_corpus` is set, `R3Dataset` loads pseudo-text by `doc_id` from the JSONL produced by `build_pseudo_text.py`, ensuring training-time retrieval uses the offline corpus.

### 8) Inference / Evaluation
- You can keep `corruptions.py` to simulate missing modality across backbones, or feed clean data; the simulator inside the model only outputs confidence maps (no extra corruption).

### 9) Key Files
- `train_r3.py` — dataset wrapper, collate, trainer, curriculum, corpus loading.
- `data_pipeline/corruptions.py` — pre-encoding occlusion/blur + pseudo-text drop.
- `r3/r3_model.py` — dual-branch forward.
- `r3/retrieval_module.py` — pseudo-text builder + noise-aware retrieval.
- `r3/reconstructor.py` — prefix/memory/imputation fusion.
- `r3/corruption_simulator.py` — confidence estimation (no token corruption).
