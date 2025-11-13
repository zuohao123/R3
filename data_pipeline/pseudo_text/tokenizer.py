"""
Pseudo-text generator that transforms multimodal signals into textual spans.
"""
from __future__ import annotations

from typing import Dict, List

# todo 如果原始数据里面没有对应的ocr信息，可能需要一个过程来进行生成

class PseudoTextTokenizer:
    def __init__(self, ocr_tag: str = "<OCR>", caption_tag: str = "<CAP>") -> None:
        self.ocr_tag = ocr_tag
        self.caption_tag = caption_tag

    def build_pseudo_text(self, sample: Dict) -> List[str]:
        pseudo_segments: List[str] = []
        ocr_tokens = sample.get("extra", {}).get("ocr_tokens", [])
        if ocr_tokens:
            pseudo_segments.append(
                f"{self.ocr_tag} " + " ".join(ocr_tokens) + f" {self.ocr_tag}"
            )

        captions = sample.get("extra", {}).get("captions", [])
        for caption in captions:
            pseudo_segments.append(f"{self.caption_tag} {caption} {self.caption_tag}")
        return pseudo_segments
