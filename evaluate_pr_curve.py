#!/usr/bin/env python
import argparse
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from acf.model import ACFDetector
from acf.inference import detect_multiscale, get_scales_octave_based
from acf.preprocessing import (
    AnnotationSetting,
    parse_wider_face_annotation,
    load_image,
    compute_iou_batch,
)


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
    precisions = [0.0] + precisions + [0.0]

    precisions = interpolate_precision(precisions)

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += precisions[i] * (recalls[i] - recalls[i - 1])

    return ap


def main():
    parser = argparse.ArgumentParser(
        description="PR-curve evaluation for ACF face detector"
    )

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--annotation_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)

    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--nms_threshold", type=float, default=0.3)

    parser.add_argument(
        "--iou_thresholds", type=float, nargs="+", default=[0.5, 0.75, 0.9, 0.95, 0.99]
    )

    parser.add_argument("--n_per_oct", type=int, default=8)
    parser.add_argument("--n_oct_up", type=int, default=2)
    parser.add_argument("--min_ds", type=int, nargs=2, default=[24, 24])
    parser.add_argument("--max_scale", type=float, default=None)

    parser.add_argument("--output", type=str, default="pr_curve.png")
    parser.add_argument("--max_images", type=int, default=None)

    args = parser.parse_args()

    detector = ACFDetector()
    detector.load(args.model)

    annotations = parse_wider_face_annotation(
        args.annotation_file,
        AnnotationSetting(None, None, None, None, None, True),
    )

    scales = get_scales_octave_based(
        args.n_per_oct,
        args.n_oct_up,
        tuple(args.min_ds),
        args.max_scale,
    )

    all_records_per_iou = {t: [] for t in args.iou_thresholds}
    total_gt = 0

    image_paths = list(annotations.keys())
    if args.max_images:
        image_paths = image_paths[: args.max_images]

    for img_path in tqdm(image_paths, desc="Evaluating", ncols=100):
        image = load_image(img_path, args.image_dir)
        gt_boxes = annotations[img_path]
        total_gt += len(gt_boxes)

        detections = detect_multiscale(
            detector=detector,
            image=image,
            scales=scales,
            stride=args.stride,
            score_threshold=0.0,
            nms_threshold=args.nms_threshold,
        )

        for iou_t in args.iou_thresholds:
            records = evaluate_image_pr(
                detections,
                gt_boxes,
                iou_t,
            )
            all_records_per_iou[iou_t].extend(records)

    ap_results = {}

    for iou_t in args.iou_thresholds:
        recalls, precisions = compute_pr_curve(
            all_records_per_iou[iou_t],
            total_gt,
        )
        ap = compute_ap(recalls, precisions)
        ap_results[iou_t] = ap

        print(f"AP@{iou_t:.2f}: {ap:.4f}")

        output = f"{iou_t}_{args.output}"

        plt.figure(figsize=(6, 5))
        plt.plot(recalls, precisions)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision–Recall Curve (AP={ap:.3f})")
        plt.grid(True)
        plt.savefig(output, dpi=300, bbox_inches="tight")

        print(f"PR curve saved to {output}")


if __name__ == "__main__":
    main()
