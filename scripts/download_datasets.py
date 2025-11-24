#!/usr/bin/env python
"""
Lightweight downloader/organizer for PMC datasets (InfoVQA, MP-DocVQA, MTVQA).
Downloads from Hugging Face when repo ids are provided, otherwise stages
already-downloaded archives/directories into the expected layout.
"""
from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    snapshot_download = None


DATASETS: Dict[str, Dict] = {
    "infovqa": {
        "repo_id": "WinKawaks/InfoVQA",
        "annotations": {
            "train": ["infovqa_train.json", "infographicsVQA_train_v1.0.json", "train.json"],
            "val": ["infovqa_val.json", "infographicsVQA_val_v1.0.json", "val.json"],
            "test": ["infovqa_test.json", "infographicsVQA_test_v1.0.json", "test.json"],
        },
        "image_dirs": ["images", "train_images", "infovqa_images"],
        "image_archives": ["images.zip", "train_images.zip", "infovqa_images.zip"],
        "target_images": "images",
    },
    "mp_docvqa": {
        # MP-DocVQA often requires manual download from the RRC site; supply a repo via
        # --repo-overrides mp_docvqa=<hf_repo> or point --local-sources to extracted files.
        "repo_id": None,
        "annotations": {
            "train": ["mp_docvqa_train.json", "train.json"],
            "val": ["mp_docvqa_val.json", "val.json"],
            "test": ["mp_docvqa_test.json", "test.json"],
        },
        "image_dirs": ["images", "pages", "docs"],
        "image_archives": ["images.zip", "documents.zip", "pages.zip"],
        "target_images": "images",
    },
    "mtvqa": {
        "repo_id": "ByteDance/MTVQA",
        "annotations": {
            "train": ["mtvqa_train.json", "train.json"],
            "val": ["mtvqa_val.json", "val.json"],
            "test": ["mtvqa_test.json", "test.json"],
        },
        "image_dirs": ["images", "imgs", "frames"],
        "image_archives": ["images.zip", "imgs.zip", "frames.zip"],
        "video_dirs": ["videos"],
        "video_archives": ["videos.zip", "video.zip"],
        "target_images": "images",
        "target_videos": "videos",
    },
}


def parse_kv_pairs(pairs: Optional[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        out[key.strip()] = val.strip()
    return out


def find_first(root: Path, candidates: Iterable[str]) -> Optional[Path]:
    for cand in candidates:
        matches = list(root.rglob(cand))
        if matches:
            return matches[0]
    return None


def discover_annotation(source_root: Path, split: str) -> Optional[Path]:
    """
    Fallback: search any *.json that contains the split name (train/val/test) under root or root/data.
    """
    search_roots = [source_root, source_root / "data", source_root / "Data"]
    for base in search_roots:
        if not base.exists():
            continue
        for path in base.rglob("*.json"):
            name_lower = path.name.lower()
            if split in name_lower:
                return path
    return None


def discover_media_dir(source_root: Path, extensions: Tuple[str, ...], min_count: int = 20) -> Optional[Path]:
    """
    Fallback: find the directory that contains the most files with given extensions.
    """
    counts: Dict[Path, int] = {}
    for ext in extensions:
        for file in source_root.rglob(f"*{ext}"):
            parent = file.parent
            counts[parent] = counts.get(parent, 0) + 1
    if not counts:
        return None
    best_dir, best_count = max(counts.items(), key=lambda kv: kv[1])
    if best_count < min_count:
        return None
    return best_dir


def discover_archive(source_root: Path, extensions: Tuple[str, ...]) -> Optional[Path]:
    for ext in extensions:
        matches = list(source_root.rglob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def sync_tree(src: Path, dst: Path, prefer_symlink: bool, overwrite: bool) -> None:
    if dst.exists() and overwrite:
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            for child in dst.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if prefer_symlink:
        if dst.exists():
            return
        dst.symlink_to(src, target_is_directory=True)
    else:
        import shutil

        shutil.copytree(src, dst, dirs_exist_ok=True)


def extract_archive(archive: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dst)
    elif archive.suffix in {".tar", ".tgz"} or archive.name.endswith(".tar.gz"):
        mode = "r:gz" if archive.suffix in {".tgz"} or archive.name.endswith(".tar.gz") else "r:"
        with tarfile.open(archive, mode) as tf:
            tf.extractall(dst)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")


def stage_annotations(name: str, source_root: Path, target_root: Path, mapping: Dict[str, List[str]], overwrite: bool) -> None:
    for split, candidates in mapping.items():
        dest = target_root / f"{name}_{split}.json"
        if dest.exists() and not overwrite:
            continue
        found = find_first(source_root, candidates) or discover_annotation(source_root, split)
        if not found:
            print(f"  ! Missing annotations for split={split} (looked for {candidates})")
            continue
        dest.write_bytes(found.read_bytes())
        print(f"  ✓ {split} annotations -> {dest}")


def stage_modalities(
    source_root: Path,
    target_root: Path,
    target_name: str,
    dir_candidates: List[str],
    archive_candidates: List[str],
    archive_extensions: Tuple[str, ...],
    file_extensions: Tuple[str, ...],
    prefer_symlink: bool,
    overwrite: bool,
) -> None:
    dest = target_root / target_name
    if dest.exists() and dest.is_dir() and any(dest.iterdir()) and not overwrite:
        print(f"  = Reusing existing {target_name} at {dest}")
        return
    dir_match = find_first(source_root, dir_candidates)
    if dir_match and dir_match.is_dir():
        sync_tree(dir_match, dest, prefer_symlink=prefer_symlink, overwrite=overwrite)
        print(f"  ✓ {target_name} dir -> {dest} (from {dir_match})")
        return
    archive_match = find_first(source_root, archive_candidates) or discover_archive(source_root, archive_extensions)
    if archive_match:
        extract_archive(archive_match, dest, overwrite=overwrite)
        print(f"  ✓ {target_name} extracted -> {dest} (from {archive_match})")
        return
    media_dir = discover_media_dir(source_root, file_extensions)
    if media_dir:
        sync_tree(media_dir, dest, prefer_symlink=prefer_symlink, overwrite=overwrite)
        print(f"  ✓ {target_name} auto-detected dir -> {dest} (from {media_dir})")
        return
    print(f"  ! No {target_name} found (searched for dirs {dir_candidates} or archives {archive_candidates})")


def download_repo(repo_id: str, token: Optional[str], cache_dir: Optional[Path]) -> Path:
    if snapshot_download is None:
        raise ImportError("huggingface_hub is required to download datasets.")
    local_path = snapshot_download(repo_id=repo_id, repo_type="dataset", token=token, cache_dir=cache_dir)
    return Path(local_path)


def stage_dataset(
    name: str,
    cfg: Dict,
    target_root: Path,
    repo_id: Optional[str],
    local_sources: Dict[str, str],
    hf_token: Optional[str],
    hf_cache: Optional[Path],
    prefer_symlink: bool,
    overwrite: bool,
) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    source_hint = local_sources.get(name)
    source_root: Optional[Path] = Path(source_hint) if source_hint else None
    if source_root and not source_root.exists():
        print(f"[{name}] Provided local source {source_root} does not exist, ignoring.")
        source_root = None
    if source_root is None and repo_id:
        print(f"[{name}] Downloading from {repo_id} ...")
        source_root = download_repo(repo_id, hf_token, hf_cache)
    if source_root is None:
        print(f"[{name}] No download source configured. Provide --repo-overrides or --local-sources.")
        return

    print(f"[{name}] Staging data from {source_root} -> {target_root}")
    stage_annotations(name, source_root, target_root, cfg.get("annotations", {}), overwrite=overwrite)
    image_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    archive_exts = (".zip", ".tar", ".tgz", ".tar.gz")
    if cfg.get("target_images"):
        stage_modalities(
            source_root,
            target_root,
            cfg["target_images"],
            cfg.get("image_dirs", []),
            cfg.get("image_archives", []),
            archive_exts,
            image_exts,
            prefer_symlink=prefer_symlink,
            overwrite=overwrite,
        )
    if cfg.get("target_videos"):
        video_exts = (".mp4", ".avi", ".mov", ".mkv")
        stage_modalities(
            source_root,
            target_root,
            cfg["target_videos"],
            cfg.get("video_dirs", []),
            cfg.get("video_archives", []),
            archive_exts,
            video_exts,
            prefer_symlink=prefer_symlink,
            overwrite=overwrite,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/organize InfoVQA, MP-DocVQA, MTVQA.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["infovqa", "mp_docvqa", "mtvqa"],
        choices=list(DATASETS.keys()),
        help="Datasets to download/stage.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data_pipeline/data"), help="Destination root.")
    parser.add_argument("--repo-overrides", nargs="+", help="Override repo ids, e.g., mp_docvqa=user/repo")
    parser.add_argument("--local-sources", nargs="+", help="Use local sources, e.g., mp_docvqa=/path/to/files")
    parser.add_argument("--hf-token", type=str, default=None, help="Optional HF token for gated repos.")
    parser.add_argument("--hf-cache", type=Path, default=None, help="Optional cache dir for HF snapshots.")
    parser.add_argument("--prefer-symlink", action="store_true", help="Symlink instead of copying media folders.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files/folders.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_overrides = parse_kv_pairs(args.repo_overrides)
    local_sources = parse_kv_pairs(args.local_sources)

    for name in args.datasets:
        cfg = DATASETS[name]
        repo_id = repo_overrides.get(name, cfg.get("repo_id"))
        try:
            stage_dataset(
                name=name,
                cfg=cfg,
                target_root=args.output_root / name,
                repo_id=repo_id,
                local_sources=local_sources,
                hf_token=args.hf_token,
                hf_cache=args.hf_cache,
                prefer_symlink=args.prefer_symlink,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # pragma: no cover
            print(f"[{name}] Failed: {exc}")


if __name__ == "__main__":
    main()
