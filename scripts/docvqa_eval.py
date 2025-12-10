"""
DocVQA ANLS/Accuracy evaluator.

Usage:
python scripts/docvqa_eval.py --gt_path /path/to/val_groundtruth.json --pred_path preds_docvqa_val.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def levenshtein_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, char_a in enumerate(a, start=1):
        curr[0] = i
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[-1]


def anls_score(pred: str, target: str, threshold: float = 0.5) -> float:
    pred_norm = " ".join(pred.lower().strip().split())
    target_norm = " ".join(target.lower().strip().split())
    if not target_norm:
        return 1.0 if not pred_norm else 0.0
    distance = levenshtein_distance(pred_norm, target_norm)
    score = 1.0 - distance / max(len(pred_norm), len(target_norm))
    return score if score >= threshold else 0.0


def load_groundtruth(path: Path) -> Dict[str, List[str]]:
    """
    Supports common DocVQA groundtruth formats:
    - list of dicts with keys: questionId or id, and answers (list[str]) or answer (str)
    - dict id -> answers
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    gt: Dict[str, List[str]] = {}
    if isinstance(data, dict):
        # already id -> answers mapping
        for k, v in data.items():
            if isinstance(v, list):
                gt[str(k)] = [str(x) for x in v]
            else:
                gt[str(k)] = [str(v)]
        return gt
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            qid = item.get("questionId") or item.get("question_id") or item.get("id")
            if qid is None:
                continue
            answers = item.get("answers") or item.get("answer") or []
            if isinstance(answers, list):
                answers_list = [str(x) for x in answers]
            else:
                answers_list = [str(answers)]
            gt[str(qid)] = answers_list
    return gt


def load_predictions(path: Path) -> Dict[str, str]:
    preds: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj.get("id") or obj.get("questionId") or obj.get("question_id")
            pred = obj.get("prediction") or obj.get("answer") or ""
            if pid is not None:
                preds[str(pid)] = str(pred)
    return preds


def evaluate(gt: Dict[str, List[str]], preds: Dict[str, str]) -> Tuple[float, float, int]:
    total = 0
    acc = 0
    anls_sum = 0.0
    for qid, answers in gt.items():
        if qid not in preds:
            continue
        pred = preds[qid]
        total += 1
        # accuracy: exact match against any answer
        if any(pred.strip().lower() == ans.strip().lower() for ans in answers):
            acc += 1
        # anls: best match over all answers
        best = max(anls_score(pred, ans) for ans in answers)
        anls_sum += best
    acc_val = acc / max(1, total)
    anls_val = anls_sum / max(1, total)
    return acc_val, anls_val, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_path", type=Path, required=True, help="Groundtruth JSON")
    ap.add_argument("--pred_path", type=Path, required=True, help="Predictions JSONL")
    args = ap.parse_args()

    gt = load_groundtruth(args.gt_path)
    preds = load_predictions(args.pred_path)
    acc, anls, total = evaluate(gt, preds)
    print(json.dumps({"samples": total, "accuracy": round(acc, 4), "anls": round(anls, 4)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
