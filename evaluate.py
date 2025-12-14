#!/usr/bin/env python
import argparse
from tqdm import tqdm
from acf.model import ACFDetector
from acf.inference import detect_multiscale, evaluate_detections
from acf.preprocessing import AnnotationSetting, parse_wider_face_annotation, load_image


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACF face detector")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        required=True,
        help="Path to WIDER FACE annotation file",
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Path to image directory"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to evaluate (None for all)",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.25, 1.5],
        help="Scales for multi-scale detection",
    )
    parser.add_argument("--stride", type=int, default=8, help="Sliding window stride")
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.7,
        help="Minimum confidence score (default: 0.7)",
    )
    parser.add_argument(
        "--nms_threshold", type=float, default=0.3, help="NMS threshold"
    )
    parser.add_argument(
        "--iou_threshold", type=float, default=0.5, help="IoU threshold for evaluation"
    )

    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    detector = ACFDetector()
    detector.load(args.model)

    print(f"Loading annotations from {args.annotation_file}...")
    annotations = parse_wider_face_annotation(
        args.annotation_file, AnnotationSetting(None, None, None, None, None, True)
    )

    image_paths = list(annotations.keys())
    if args.max_images:
        image_paths = image_paths[: args.max_images]

    print(f"Evaluating on {len(image_paths)} images...")

    all_metrics = []

    for img_path in tqdm(image_paths, desc="Evaluating", ncols=100):
        try:
            image = load_image(img_path, args.image_dir)

            detections = detect_multiscale(
                detector=detector,
                image=image,
                scales=args.scales,
                stride=args.stride,
                score_threshold=args.score_threshold,
                nms_threshold=args.nms_threshold,
            )

            gt_boxes = annotations[img_path]

            metrics = evaluate_detections(detections, gt_boxes, args.iou_threshold)
            all_metrics.append(metrics)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    if all_metrics:
        total_tp = sum([m["true_positives"] for m in all_metrics])
        total_fp = sum([m["false_positives"] for m in all_metrics])
        total_fn = sum([m["false_negatives"] for m in all_metrics])

        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
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
