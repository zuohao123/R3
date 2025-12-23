"""
Pseudo-text guided adaptive retrieval for the R^3 architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


@dataclass
class RetrievalModuleConfig:
    hidden_size: int = 4096
    top_k: int = 3
    enable: bool = True
    cache_path: Optional[str] = None  # enable FAISS/vector store when set
    max_evidence_tokens: int = 32     # max tokens per evidence string when tokenizer is available
    external_chunk_size: int = 4096   # CPU top-k chunk size for large external corpus
    use_pseudo_query: bool = True
    pseudo_query_weight: float = 0.6


@dataclass
class PseudoTextBuilderConfig:
    """
    Controls how pseudo-text is synthesized when OCR / captions are missing.
    """

    default_conf: float = 0.75
    include_ocr: bool = True
    include_caption: bool = True
    include_context: bool = True  # keep nearby tokens for page-as-evidence


class PseudoTextBuilder:
    """
    Normalizes OCR / caption / fallback text into retrieval-ready strings.
    优先级：context_evidence（跨页/外部） > OCR > Caption > Fallback(Q/ID)
    """

    def __init__(self, config: PseudoTextBuilderConfig | None = None, fallback_caption_fn=None) -> None:
        self.config = config or PseudoTextBuilderConfig()
        self.fallback_caption_fn = fallback_caption_fn

    def build(self, sample: Dict) -> List[str]:
        extra = sample.get("extra", {}) or {}
        entries: List[str] = []
        # Context evidence (highest priority for Page-as-Evidence)
        if self.config.include_context:
            for ctx in extra.get("context_evidence", []) or []:
                if ctx:
                    entries.append(str(ctx))
        # OCR tokens
        if self.config.include_ocr:
            for token in extra.get("ocr_tokens", []) or []:
                text = token.get("text") if isinstance(token, dict) else str(token)
                if text:
                    entries.append(text)
        # Captions (existing or on-the-fly fallback)
        if self.config.include_caption:
            for caption in extra.get("captions", []) or []:
                if caption:
                    entries.append(str(caption))
            if not extra.get("captions") and self.fallback_caption_fn and sample.get("image_path"):
                caption = self.fallback_caption_fn(sample["image_path"])
                if caption:
                    entries.append(str(caption))
        # Optional contextual evidence (page-as-evidence)
        if self.config.include_context:
            for ctx in extra.get("context_evidence", []) or []:
                if ctx:
                    entries.append(str(ctx))

        # If nothing exists, fall back to question / id anchors to avoid empty retrieval sets.
        if not entries:
            question = sample.get("question", "")
            doc_id = sample.get("id", "")
            if question:
                entries.append(f"[Q] {question}")
            if doc_id:
                entries.append(f"[ID] {doc_id}")
        return entries


class PseudoTextRetrievalModule(nn.Module):
    """
    Lightweight retrieval built on pseudo-text tokens.
    支持两种来源：
      1) 批内伪文本（默认）
      2) ingest_corpus 预载的外部伪文本库（build_pseudo_text 生成的 JSONL）
    支持两种后端：
      - 内存 hashing + 余弦
      - 可选 FAISS（cache_path 且安装 faiss 时）
    """

    def __init__(self, config: RetrievalModuleConfig, embedding_layer: nn.Embedding, tokenizer=None) -> None:
        super().__init__()
        self.config = config
        # NOTE: The embedding layer is owned by the backbone LM. Do NOT register it as a child module here,
        # otherwise it appears twice in the top-level state_dict (shared tensor), and safetensors will
        # refuse to save checkpoints. Keep a plain reference instead.
        # Bypass nn.Module.__setattr__ so it won't be tracked in `self._modules`.
        self.__dict__["_embedding_layer"] = embedding_layer
        self.__dict__["_tokenizer"] = tokenizer
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.evidence_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.scorer = nn.Linear(config.hidden_size, 1)
        self.use_faiss = bool(config.cache_path and faiss is not None)
        self.index = None
        if self.use_faiss:
            self._init_faiss_index(config.hidden_size)
        self._faiss_ids: List[str] = []
        self.external_embeddings: Optional[torch.Tensor] = None
        self.external_texts: Optional[List[str]] = None

    def forward(
        self,
        question_embeddings: torch.Tensor,
        pseudo_text: Sequence[Sequence[str]],
        img_conf: torch.Tensor,
        txt_conf: torch.Tensor,
    ) -> Dict[str, torch.Tensor | List[List[str]]]:
        if not self.config.enable:
            batch_size = question_embeddings.size(0)
            return {
                "texts": [[] for _ in range(batch_size)],
                "embeddings": torch.zeros(
                    batch_size,
                    0,
                    1,
                    self.config.hidden_size,
                    device=question_embeddings.device,
                    dtype=torch.float32,
                ),
                "scores": torch.zeros(batch_size, 0, device=question_embeddings.device, dtype=torch.float32),
            }

        # Keep retrieval math in fp32 for stability; cast inputs as needed.
        device = self.query_proj.weight.device
        query = self._build_query(
            question_embeddings.to(device=device).float(),
            txt_conf.to(device=device).float(),
        )  # (b, d)
        pseudo_query = None
        evidence_embeddings = None
        evidence_texts: Optional[List[List[str]]] = None
        if self.config.use_pseudo_query:
            pseudo_embeddings, pseudo_texts = self._encode_evidence(pseudo_text, device)
            pseudo_query = self._build_pseudo_query(pseudo_embeddings).to(dtype=query.dtype)
            evidence_embeddings = pseudo_embeddings
            evidence_texts = pseudo_texts
        if self.external_embeddings is not None and self.external_texts is not None:
            # Use external corpus built from build_pseudo_text.py outputs
            evidence_embeddings = self.external_embeddings
            evidence_texts = [self.external_texts for _ in range(question_embeddings.size(0))]
        elif evidence_embeddings is None or evidence_texts is None:
            evidence_embeddings, evidence_texts = self._encode_evidence(pseudo_text, device)
        evidence_embeddings = evidence_embeddings.float()
        if pseudo_query is not None:
            weight = float(self.config.pseudo_query_weight)
            weight = max(0.0, min(1.0, weight))
            query = (1.0 - weight) * query + weight * pseudo_query

        if (
            self.external_embeddings is not None
            and self.external_texts is not None
            and evidence_embeddings.device.type == "cpu"
            and device.type == "cuda"
        ):
            topk_embeddings, topk_texts, topk_scores = self._cpu_topk_external(
                query.detach().to("cpu"),
                evidence_embeddings,
                evidence_texts,
                top_k=self.config.top_k,
                chunk_size=self.config.external_chunk_size,
            )
            # Move selected evidence to GPU for downstream reconstruction.
            topk_embeddings = topk_embeddings.to(device=device).float()
            topk_scores = topk_scores.to(device=device).float()
        elif self.use_faiss and self.index is not None:
            topk_embeddings, topk_texts, topk_scores = self._faiss_search(evidence_embeddings, evidence_texts, query)
        else:
            evidence_embeddings = evidence_embeddings.to(device=device).float()
            scores = self._score(
                query,
                evidence_embeddings,
                img_conf.to(device=device).float(),
                txt_conf.to(device=device).float(),
            )
            topk_embeddings, topk_texts, topk_scores = self._select_topk(evidence_embeddings, evidence_texts, scores)
        return {
            "texts": topk_texts,
            "embeddings": topk_embeddings,
            "scores": topk_scores,
        }

    def _build_pseudo_query(self, pseudo_embeddings: torch.Tensor) -> torch.Tensor:
        pooled = pseudo_embeddings.squeeze(2).mean(dim=1)
        target_dtype = self.query_proj.weight.dtype
        pooled = pooled.to(dtype=target_dtype, device=self.query_proj.weight.device)
        return self.query_proj(pooled)

    def _cpu_topk_external(
        self,
        query: torch.Tensor,
        evidence_embeddings: torch.Tensor,
        evidence_texts: List[List[str]],
        top_k: int,
        chunk_size: int,
    ) -> Tuple[torch.Tensor, List[List[str]], torch.Tensor]:
        # evidence_embeddings: (1, E, 1, D) on CPU
        with torch.no_grad():
            evidence = evidence_embeddings.squeeze(0).squeeze(1)  # (E, D)
            if evidence.dim() != 2:
                raise ValueError("External evidence embeddings must be 2D after squeeze.")
            num_items = evidence.size(0)
            batch = query.size(0)
            top_k = min(top_k, num_items) if num_items > 0 else 0
            top_scores = torch.full((batch, top_k), -1e9, dtype=torch.float32)
            top_indices = torch.full((batch, top_k), -1, dtype=torch.long)
            if num_items == 0 or top_k == 0:
                empty_embeds = torch.zeros(batch, 0, 1, self.config.hidden_size, dtype=torch.float32)
                empty_scores = torch.zeros(batch, 0, dtype=torch.float32)
                return empty_embeds, [[] for _ in range(batch)], empty_scores
            for start in range(0, num_items, chunk_size):
                chunk = evidence[start : start + chunk_size].float()  # (C, D)
                scores = torch.matmul(query.float(), chunk.t())  # (B, C)
                c = scores.size(1)
                idx_chunk = torch.arange(start, start + c).unsqueeze(0).expand(batch, c)
                merged_scores = torch.cat([top_scores, scores], dim=1)
                merged_indices = torch.cat([top_indices, idx_chunk], dim=1)
                new_scores, new_pos = torch.topk(merged_scores, k=top_k, dim=1)
                top_scores = new_scores
                top_indices = merged_indices.gather(1, new_pos)
            gathered = evidence[top_indices]  # (B, K, D)
            top_embeddings = gathered.unsqueeze(2)  # (B, K, 1, D)
            texts = []
            flat_texts = evidence_texts[0] if evidence_texts else []
            for b in range(batch):
                texts.append([flat_texts[i] if 0 <= i < len(flat_texts) else "" for i in top_indices[b].tolist()])
            return top_embeddings, texts, top_scores

    def _build_query(self, question_embeddings: torch.Tensor, txt_conf: torch.Tensor) -> torch.Tensor:
        weights = (1.0 - txt_conf).unsqueeze(-1)
        pooled = (question_embeddings * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-6)
        pooled = pooled.to(dtype=self.query_proj.weight.dtype, device=self.query_proj.weight.device)
        return self.query_proj(pooled)

    def _encode_evidence(
        self,
        pseudo_text: Sequence[Sequence[str]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, List[List[str]]]:
        embeddings = []
        texts: List[List[str]] = []
        for entries in pseudo_text:
            encoded_entries = []
            stored_texts = []
            for text in entries:
                if not text:
                    continue
                vec = None
                tokenizer = getattr(self, "_tokenizer", None)
                if tokenizer is not None:
                    try:
                        tokenized = tokenizer(
                            text,
                            add_special_tokens=False,
                            truncation=True,
                            max_length=self.config.max_evidence_tokens,
                            return_tensors="pt",
                        )
                        input_ids = tokenized["input_ids"].to(device=device)
                        if input_ids.numel() > 0:
                            vec = self._embedding_layer(input_ids).mean(dim=1).squeeze(0)
                    except Exception:
                        vec = None
                if vec is None:
                    tokens = torch.tensor(
                        [hash(text) % self._embedding_layer.num_embeddings],
                        device=device,
                    )
                    vec = self._embedding_layer(tokens).mean(dim=0)
                encoded_entries.append(vec)
                stored_texts.append(text)
            if not encoded_entries:
                encoded_entries = [
                    torch.zeros(
                        self.config.hidden_size,
                        device=device,
                        dtype=self._embedding_layer.weight.dtype,
                    )
                ]
                stored_texts.append("")
            embeddings.append(torch.stack(encoded_entries))
            texts.append(stored_texts)
        max_len = max(e.size(0) for e in embeddings)
        padded = []
        for emb in embeddings:
            if emb.size(0) < max_len:
                pad = torch.zeros(max_len - emb.size(0), emb.size(1), device=device, dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=0)
            padded.append(emb)
        stacked = torch.stack(padded)  # (b, evidences, d)
        return stacked.unsqueeze(2), texts  # reshape to (b, evidences, 1, d) for downstream

    def _faiss_search(
        self,
        embeddings: torch.Tensor,
        texts: List[List[str]],
        query: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[List[str]], torch.Tensor]:
        """
        Build/update a FAISS index for evidence and retrieve top-k per query.
        This is a lightweight adapter; for large corpora use an offline index.
        """
        bsz, evidences, _, dim = embeddings.shape
        batch_embeddings = []
        batch_texts: List[List[str]] = []
        batch_scores = []
        for b in range(bsz):
            flat = F.normalize(embeddings[b].squeeze(1), dim=-1).detach().cpu().contiguous()
            index = faiss.IndexFlatIP(dim)
            index.add(flat.numpy())
            query_cpu = F.normalize(query[b:b+1], dim=-1).detach().cpu().numpy()
            scores, idx = index.search(query_cpu, min(self.config.top_k, evidences))
            sel = idx[0]
            emb = embeddings[b, sel]
            batch_embeddings.append(emb)
            batch_texts.append([texts[b][i] if i < len(texts[b]) else "" for i in sel])
            batch_scores.append(torch.tensor(scores[0], device=embeddings.device))
        return torch.stack(batch_embeddings), batch_texts, torch.stack(batch_scores)

    def _init_faiss_index(self, dim: int) -> None:
        if faiss is None:
            self.use_faiss = False
            return
        self.index = faiss.IndexFlatIP(dim)

    def _score(
        self,
        query: torch.Tensor,
        evidence: torch.Tensor,
        img_conf: torch.Tensor,
        txt_conf: torch.Tensor,
    ) -> torch.Tensor:
        evidence = evidence.to(dtype=self.evidence_proj.weight.dtype, device=self.evidence_proj.weight.device)
        proj = self.evidence_proj(evidence).squeeze(2)  # (b, evidences, d)
        query = query.unsqueeze(1)
        logits = torch.cosine_similarity(query, proj, dim=-1)

        # Noise-aware boosting: low-visibility regions should rely more on textual evidence.
        mask_intensity = 1.0 - img_conf  # (b, img_tokens)
        visual_uncertainty = mask_intensity.mean(dim=1, keepdim=True)  # coarse spatial proxy
        text_uncertainty = (1.0 - txt_conf).mean(dim=1, keepdim=True)
        noise_gate = 1.0 + visual_uncertainty  # boost when vision is unreliable
        attenuation = 1.0 - 0.5 * text_uncertainty  # but do not over-trust noisy text
        return logits * noise_gate * attenuation

    def ingest_corpus(self, corpus_path: Optional[str]) -> None:
        """
        Load an external pseudo-text corpus (JSONL with `pseudo_text` field) and
        pre-embed for retrieval. Intended to consume build_pseudo_text.py outputs.
        加载后 forward 将优先使用外部库，覆盖批内伪文本。
        """
        if not corpus_path:
            return
        path = Path(corpus_path)
        if not path.exists():
            return
        texts: List[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    import json
                    obj = json.loads(line)
                    texts.extend(obj.get("pseudo_text", []))
                except Exception:
                    continue
        unique_texts = [t for t in texts if t]
        # If the corpus is token-level OCR (lots of single-word entries), coalesce into chunks.
        if unique_texts:
            word_counts = [len(str(t).split()) for t in unique_texts if str(t).strip()]
            if word_counts:
                avg_words = sum(word_counts) / len(word_counts)
                short_ratio = sum(1 for c in word_counts if c <= 1) / len(word_counts)
                if len(unique_texts) >= 40 and avg_words < 1.5 and short_ratio > 0.8:
                    tokens: List[str] = []
                    for t in unique_texts:
                        text = str(t).strip()
                        if not text:
                            continue
                        tokens.extend(text.split())
                    chunks: List[str] = []
                    for idx in range(0, len(tokens), 12):
                        chunk = " ".join(tokens[idx : idx + 12]).strip()
                        if chunk:
                            chunks.append(chunk)
                    unique_texts = chunks
        if not unique_texts:
            return
        device = next(self._embedding_layer.parameters()).device
        vectors = []
        for text in unique_texts:
            vec = None
            tokenizer = getattr(self, "_tokenizer", None)
            if tokenizer is not None:
                try:
                    tokenized = tokenizer(
                        text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.config.max_evidence_tokens,
                        return_tensors="pt",
                    )
                    input_ids = tokenized["input_ids"].to(device=device)
                    if input_ids.numel() > 0:
                        vec = self._embedding_layer(input_ids).mean(dim=1).squeeze(0)
                except Exception:
                    vec = None
            if vec is None:
                token = torch.tensor([hash(text) % self._embedding_layer.num_embeddings], device=device)
                vec = self._embedding_layer(token).mean(dim=0)
            vectors.append(vec.detach().cpu())
        self.external_embeddings = torch.stack(vectors).unsqueeze(0).unsqueeze(2)  # CPU tensor
        self.external_texts = unique_texts

    def _select_topk(
        self,
        embeddings: torch.Tensor,
        texts: List[List[str]],
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[List[str]], torch.Tensor]:
        k = min(self.config.top_k, embeddings.size(1))
        topk_scores, topk_indices = torch.topk(scores, k=k, dim=1)
        batch_embeddings = []
        batch_texts: List[List[str]] = []
        for b in range(embeddings.size(0)):
            indices = topk_indices[b]
            batch_embeddings.append(embeddings[b, indices])
            batch_texts.append([texts[b][idx] if idx < len(texts[b]) else "" for idx in indices])
        stacked_embeddings = torch.stack(batch_embeddings)
        return stacked_embeddings, batch_texts, topk_scores
