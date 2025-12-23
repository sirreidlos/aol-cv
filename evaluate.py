#!/usr/bin/env python
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Literal
from tqdm import tqdm
from acf.model import ACFDetector
from acf.inference import (
    detect_multiscale,
    evaluate_detections,
    get_scales_octave_based,
)
from acf.preprocessing import (
    AnnotationSetting,
    parse_wider_face_annotation,
    load_image,
    get_muct_annotations,
)


@dataclass
class Args:
    model: str
    annotation_file: str
    image_dir: str
    dataset: Literal["widerface", "muct"]
    max_images: Optional[int]

    stride: int
    score_threshold: float
    nms_threshold: float
    iou_threshold: float

    n_per_oct: int
    n_oct_up: int
    max_scale: Optional[float]
    min_ds: Tuple[int, int]
    max_ds: Tuple[int, int]


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Evaluate ACF face detector")

    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--annotation_file", type=str, required=True, help="Path to annotation file"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Path to image directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="widerface",
        choices=["widerface", "muct"],
        help="Dataset type",
    )
    parser.add_argument("--max_images", type=int, default=None)

    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.3)
    parser.add_argument("--iou_threshold", type=float, default=0.5)

    parser.add_argument("--n_per_oct", type=int, default=8)
    parser.add_argument("--n_oct_up", type=int, default=2)
    parser.add_argument("--max_scale", type=float, default=None)
    parser.add_argument("--min_ds", type=int, nargs=2, default=[24, 24])
    parser.add_argument("--max_ds", type=int, nargs=2, default=[256, 256])

    args = parser.parse_args()
    args.min_ds = tuple(args.min_ds)
    return Args(**vars(args))


def main():
    args = parse_args()

    print(f"Loading model from {args.model}...")
    detector = ACFDetector()
    detector.load(args.model)

    print("Preparing scales for multiscale detection...")
    scales = get_scales_octave_based(
        args.n_per_oct, args.n_oct_up, args.min_ds, args.max_scale
    )

    if args.dataset == "widerface":
        print(f"Loading WIDER FACE annotations from {args.annotation_file}...")
        annotations = parse_wider_face_annotation(
            args.annotation_file, AnnotationSetting(True)
        )
    elif args.dataset == "muct":
        print(f"Loading MUCT annotations from {args.annotation_file}...")
        _, annotations = get_muct_annotations(args.annotation_file, args.image_dir)

    image_paths = list(annotations.keys())
    if args.max_images:
        image_paths = image_paths[: args.max_images]

    print(f"Evaluating on {len(image_paths)} images...")
    all_metrics = []

    processed = 0
    for img_path in tqdm(image_paths, desc="Evaluating", ncols=100):
        try:
            processed += 1
            image = load_image(img_path, args.image_dir)
            detections = detect_multiscale(
                detector=detector,
                image=image,
                window_size=args.max_ds,
                scales=scales,
                stride=args.stride,
                score_threshold=args.score_threshold,
                nms_threshold=args.nms_threshold,
            )
            gt_boxes = annotations[img_path]
            metrics = evaluate_detections(detections, gt_boxes, args.iou_threshold)
            all_metrics.append(metrics)

            if len(detections) == 0:
                print(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    if all_metrics:
        total_tp = sum(m["true_positives"] for m in all_metrics)
        total_fp = sum(m["false_positives"] for m in all_metrics)
        total_fn = sum(m["false_negatives"] for m in all_metrics)

        accuracy = total_tp / (total_tp + total_fp + total_fn + 1e-6)
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"\nTotal True Positives:  {total_tp}")
        print(f"Total False Positives: {total_fp}")
        print(f"Total False Negatives: {total_fn}")
        print("=" * 50)
    else:
        print("No evaluation results available")


if __name__ == "__main__":
    main()
