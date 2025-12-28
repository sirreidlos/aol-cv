#!/usr/bin/env python
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Literal, Dict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import hashlib

from acf.model import ACFDetector
from acf.inference import detect_multiscale, get_scales_octave_based
from acf.preprocessing import (
    AnnotationSetting,
    parse_wider_face_annotation,
    load_image,
    compute_iou_batch,
    get_muct_annotations,
)


@dataclass(frozen=True)
class EvalConfig:
    model: str
    annotation_file: str
    image_dir: str
    dataset: str
    max_images: Optional[int]
    stride: int
    nms_threshold: float
    batch_size: int
    iou_thresholds: Tuple[float, ...]
    n_per_oct: int
    n_oct_up: int
    max_scale: Optional[float]
    min_ds: Tuple[int, int]
    max_ds: Tuple[int, int]

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class Args:
    models: List[str]
    annotation_file: str
    image_dir: str
    dataset: Literal["widerface", "muct"]
    max_images: Optional[int]

    stride: int
    nms_threshold: float
    batch_size: int

    iou_thresholds: List[float]

    n_per_oct: int
    n_oct_up: int
    max_scale: Optional[float]
    min_ds: Tuple[int, int]
    max_ds: Tuple[int, int]

    output: str
    cache_dir: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="PR-curve evaluation for ACF face detector"
    )
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument("--annotation_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--dataset", choices=["widerface", "muct"], default="widerface")
    parser.add_argument("--max_images", type=int, default=None)

    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--nms_threshold", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--iou_thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 0.9],
    )

    parser.add_argument("--n_per_oct", type=int, default=8)
    parser.add_argument("--n_oct_up", type=int, default=2)
    parser.add_argument("--max_scale", type=float, default=None)
    parser.add_argument("--min_ds", type=int, nargs=2, default=[24, 24])
    parser.add_argument("--max_ds", type=int, nargs=2, default=[256, 256])

    parser.add_argument("--output", type=str, default="pr_curve.png")
    parser.add_argument("--cache_dir", type=str, default=".eval_cache")

    args = parser.parse_args()
    args.min_ds = tuple(args.min_ds)
    args.max_ds = tuple(args.max_ds)

    return Args(**vars(args))


def evaluate_image_pr(
    detections: List[Tuple[int, int, int, int, float]],
    gt_boxes: List[Tuple[int, int, int, int]] | np.ndarray,
    iou_threshold: float,
) -> List[Tuple[float, int]]:
    detections = sorted(detections, key=lambda d: d[4], reverse=True)

    gt_used = [False] * len(gt_boxes)
    records = []

    det_windows = np.array([d[:4] for d in detections], dtype=np.float32)
    det_scores = [d[4] for d in detections]

    iou_matrix = compute_iou_batch(det_windows, gt_boxes)

    for det_idx, score in enumerate(det_scores):
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx in range(len(gt_boxes)):
            if gt_used[gt_idx]:
                continue

            iou = iou_matrix[det_idx, gt_idx]
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            gt_used[best_gt_idx] = True
            records.append((score, 1))
        else:
            records.append((score, 0))

    return records


def compute_pr_curve(
    records: List[Tuple[float, int]], total_gt: int
) -> Tuple[List[float], List[float]]:
    records.sort(key=lambda r: r[0], reverse=True)

    tp = 0
    fp = 0

    precisions = []
    recalls = []

    for _, is_tp in records:
        if is_tp:
            tp += 1
        else:
            fp += 1

        precision = tp / (tp + fp)
        recall = tp / total_gt if total_gt > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    return recalls, precisions


def interpolate_precision(precisions: List[float]) -> List[float]:
    interp = precisions.copy()
    for i in range(len(interp) - 2, -1, -1):
        interp[i] = max(interp[i], interp[i + 1])
    return interp


def compute_ap(recalls: List[float], precisions: List[float]) -> float:
    if not recalls:
        return 0.0

    recalls = [0.0] + recalls + [1.0]
    precisions = [1.0] + precisions + [0.0]

    precisions = interpolate_precision(precisions)

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += precisions[i] * (recalls[i] - recalls[i - 1])

    return ap


def load_or_run_eval(cfg: EvalConfig, args: Args) -> Dict:
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, f"{cfg.hash()}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data["result"].item()

    detector = ACFDetector()
    detector.load(cfg.model)

    scales = get_scales_octave_based(cfg.n_per_oct, cfg.n_oct_up, cfg.max_scale)

    if cfg.dataset == "widerface":
        annotations = parse_wider_face_annotation(
            cfg.annotation_file, AnnotationSetting(True)
        )
    else:
        _, annotations = get_muct_annotations(cfg.annotation_file, cfg.image_dir)

    image_paths = list(annotations.keys())
    if cfg.max_images:
        image_paths = image_paths[: cfg.max_images]

    total_gt = 0
    records_per_iou = {t: [] for t in cfg.iou_thresholds}

    for img_path in tqdm(image_paths, desc=os.path.basename(cfg.model)):
        img = load_image(img_path, cfg.image_dir)
        gt = annotations[img_path]
        total_gt += len(gt)

        dets = detect_multiscale(
            detector=detector,
            image=img,
            window_size=cfg.max_ds,
            scales=scales,
            stride=cfg.stride,
            score_threshold=0.0,
            batch_size=cfg.batch_size,
            nms_threshold=cfg.nms_threshold,
        )

        for t in cfg.iou_thresholds:
            records_per_iou[t].extend(evaluate_image_pr(dets, gt, t))

    result = {
        "total_gt": total_gt,
        "records": records_per_iou,
    }
    np.savez_compressed(cache_path, result=result)
    return result


def main():
    args = parse_args()

    for iou_t in args.iou_thresholds:
        plt.figure(figsize=(6, 5))

        for model in args.models:
            cfg = EvalConfig(
                model=model,
                annotation_file=args.annotation_file,
                image_dir=args.image_dir,
                dataset=args.dataset,
                max_images=args.max_images,
                stride=args.stride,
                nms_threshold=args.nms_threshold,
                batch_size=args.batch_size,
                iou_thresholds=tuple(args.iou_thresholds),
                n_per_oct=args.n_per_oct,
                n_oct_up=args.n_oct_up,
                max_scale=args.max_scale,
                min_ds=args.min_ds,
                max_ds=args.max_ds,
            )

            result = load_or_run_eval(cfg, args)
            recalls, precisions = compute_pr_curve(
                result["records"][iou_t],
                result["total_gt"],
            )
            ap = compute_ap(recalls, precisions)

            label = f"{os.path.basename(model)} (AP={ap:.3f})"
            plt.plot(recalls, precisions, label=label)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve @ IoU={iou_t}")
        plt.legend()
        plt.grid(True)

        out = f"iou_{iou_t}_{args.output}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
