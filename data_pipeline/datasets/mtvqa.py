"""
MTVQA dataset adapter for R^3.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import json

from .base_dataset import BasePMCDataset


class MTVQADataset(BasePMCDataset):
    """
    Expects mtvqa_{split}.json with fields:
    [
      {
        "id": "...",
        "question": "...",
        "answer": "...",
        "image": "path or url",
        "video": "path or url",  # 可选，视频文件
        "ocr_tokens": [...],
        "captions": [...],
        "video_frames": [...],   # 可选，视频帧信息
        "temporal_info": {...}   # 可选，时序信息
      }
    ]
    """

    def __init__(self, root: Path, split: str = "train") -> None:
        self.image_root = root / "images"
        self.video_root = root / "videos"  # 视频文件目录
        super().__init__(root, split)

    def _build_index(self) -> List[Dict]:
        annot_path = self.root / f"mtvqa_{self.split}.json"
        if not annot_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {annot_path}")
        with annot_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError("MTVQA annotation must be a list.")
        normalized: List[Dict] = []
        for idx, entry in enumerate(payload):
            sample = dict(entry)
            sid = entry.get("id") or f"{self.split}_{idx}"
            sample["id"] = sid
            # Fallback: if question/answer empty, try qa_pairs
            if (not sample.get("question")) or (not sample.get("answer")):
                qa_raw = entry.get("qa_pairs")
                qa_list = qa_raw
                if isinstance(qa_raw, str):
                    try:
                        qa_list = json.loads(qa_raw)
                    except Exception:
                        qa_list = []
                if isinstance(qa_list, list) and qa_list:
                    sample.setdefault("question", qa_list[0].get("question", ""))
                    first_ans = qa_list[0].get("answer", "")
                    if isinstance(first_ans, list):
                        first_ans = first_ans[0] if first_ans else ""
                    sample.setdefault("answer", first_ans)
            normalized.append(sample)
        return normalized

    def _load_raw_item(self, sample_meta: Dict) -> Dict:
        # 处理图像路径
        image_candidate = sample_meta.get("image") or sample_meta.get("image_path") or sample_meta.get("image_id")
        image_path = self._resolve_image_path(str(image_candidate or sample_meta["id"]))
        
        # 处理视频路径（如果存在）
        video_candidate = sample_meta.get("video") or sample_meta.get("video_path") or sample_meta.get("video_id")
        video_path = self._resolve_video_path(str(video_candidate or "")) if video_candidate else None
        
        extra = {
            "ocr_tokens": self._normalize_ocr_tokens(sample_meta.get("ocr_tokens") or []),
            "captions": sample_meta.get("captions", []),
            "context_evidence": sample_meta.get("context_evidence", []),
            "video_frames": sample_meta.get("video_frames", []),  # 视频帧信息
            "temporal_info": sample_meta.get("temporal_info", {}),  # 时序信息
            "metadata": {
                "dataset": "MTVQA",
                "split": self.split,
                "has_video": video_path is not None,
                "video_path": video_path,
            },
        }
        return {
            "question": sample_meta["question"],
            "answer": sample_meta.get("answer"),
            "image_path": image_path,
            "video_path": video_path,  # 添加视频路径
            "extra": extra,
        }

    def _resolve_image_path(self, identifier: str) -> str:
        if not identifier:
            return identifier
        path_obj = Path(identifier)
        candidates = [path_obj.name]
        if not path_obj.suffix:
            candidates.extend([f"{path_obj.name}.png", f"{path_obj.name}.jpg"])
        for cand in candidates:
            resolved = self.image_root / cand
            if resolved.exists():
                return resolved.as_posix()
        return identifier

    def _resolve_video_path(self, identifier: str) -> str:
        """解析视频文件路径"""
        if not identifier:
            return identifier
        path_obj = Path(identifier)
        candidates = [path_obj.name]
        if not path_obj.suffix:
            candidates.extend([f"{path_obj.name}.mp4", f"{path_obj.name}.avi", f"{path_obj.name}.mov"])
        for cand in candidates:
            resolved = self.video_root / cand
            if resolved.exists():
                return resolved.as_posix()
        return identifier

    def _normalize_ocr_tokens(self, tokens: List) -> List[Dict]:
        normalized: List[Dict] = []
        for token in tokens:
            if isinstance(token, str):
                normalized.append({"text": token, "bbox": [0, 0, 0, 0], "conf": 1.0, "src": "ocr"})
            elif isinstance(token, dict):
                normalized.append(
                    {
                        "text": token.get("text", ""),
                        "bbox": token.get("bbox", [0, 0, 0, 0]),
                        "conf": token.get("conf", 1.0),
                        "src": token.get("src", "ocr"),
                        "frame_idx": token.get("frame_idx", 0),  # 视频帧索引
                        "timestamp": token.get("timestamp", 0.0),  # 时间戳
                    }
                )
        return normalized
