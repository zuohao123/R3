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

    def __init__(self, config: RetrievalModuleConfig, embedding_layer: nn.Embedding) -> None:
        super().__init__()
        self.config = config
        self.embedding_layer = embedding_layer
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
                "embeddings": torch.zeros(batch_size, 0, 1, self.config.hidden_size, device=question_embeddings.device),
                "scores": torch.zeros(batch_size, 0, device=question_embeddings.device),
            }

        query = self._build_query(question_embeddings, txt_conf)  # (b, d)
        if self.external_embeddings is not None and self.external_texts is not None:
            # Use external corpus built from build_pseudo_text.py outputs
            evidence_embeddings = self.external_embeddings.to(question_embeddings.device)
            evidence_texts = [self.external_texts for _ in range(question_embeddings.size(0))]
        else:
            evidence_embeddings, evidence_texts = self._encode_evidence(pseudo_text, question_embeddings.device)

        if self.use_faiss and self.index is not None:
            topk_embeddings, topk_texts, topk_scores = self._faiss_search(evidence_embeddings, evidence_texts, query)
        else:
            scores = self._score(query, evidence_embeddings, img_conf, txt_conf)
            topk_embeddings, topk_texts, topk_scores = self._select_topk(evidence_embeddings, evidence_texts, scores)
        return {
            "texts": topk_texts,
            "embeddings": topk_embeddings,
            "scores": topk_scores,
        }

    def _build_query(self, question_embeddings: torch.Tensor, txt_conf: torch.Tensor) -> torch.Tensor:
        weights = (1.0 - txt_conf).unsqueeze(-1)
        pooled = (question_embeddings * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-6)
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
                tokens = torch.tensor(
                    [hash(text) % self.embedding_layer.num_embeddings],
                    device=device,
                )
                vec = self.embedding_layer(tokens)
                encoded_entries.append(vec.mean(dim=0))
                stored_texts.append(text)
            if not encoded_entries:
                encoded_entries = [torch.zeros(self.config.hidden_size, device=device)]
                stored_texts.append("")
            embeddings.append(torch.stack(encoded_entries))
            texts.append(stored_texts)
        max_len = max(e.size(0) for e in embeddings)
        padded = []
        for emb in embeddings:
            if emb.size(0) < max_len:
                pad = torch.zeros(max_len - emb.size(0), emb.size(1), device=device)
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
        if not unique_texts:
            return
        device = next(self.embedding_layer.parameters()).device
        vectors = []
        for text in unique_texts:
            token = torch.tensor([hash(text) % self.embedding_layer.num_embeddings], device=device)
            vec = self.embedding_layer(token).mean(dim=0)
            vectors.append(vec)
        self.external_embeddings = torch.stack(vectors).unsqueeze(0).unsqueeze(2)  # (1, evidences, 1, dim)
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
