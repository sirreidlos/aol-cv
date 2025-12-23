#!/usr/bin/env python
import argparse
import cv2
import shutil
from pathlib import Path
from acf.model import ACFDetector
from acf.inference import (
    detect_multiscale,
    get_scales_octave_based,
    visualize_detections,
    visualize_feature_map,
)
from acf.preprocessing import load_image


def prepare_output_dir(output_dir):
    """Create output directory, clearing it if it already exists."""
    path = Path(output_dir)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Run ACF face detection with feature visualization"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output detection image (optional)",
    )
    parser.add_argument(
        "--feature_vis_dir",
        type=str,
        default=None,
        help="Directory to save feature channel visualizations",
    )
    parser.add_argument("--stride", type=int, default=8, help="Sliding window stride")
    parser.add_argument(
        "--score_threshold", type=float, default=0.5, help="Minimum confidence score"
    )
    parser.add_argument(
        "--nms_threshold", type=float, default=0.3, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--use_fast_pyramid",
        action="store_true",
        default=True,
        help="Use fast feature pyramid (default: True)",
    )
    parser.add_argument(
        "--no_fast_pyramid",
        dest="use_fast_pyramid",
        action="store_false",
        help="Use original channel pyramid instead of fast pyramid",
    )
    parser.add_argument(
        "--n_per_oct",
        type=int,
        default=8,
        help="Number of scales per octave for octave-based scaling",
    )
    parser.add_argument(
        "--n_oct_up",
        type=int,
        default=2,
        help="Number of octaves up for octave-based scaling",
    )
    parser.add_argument(
        "--min_ds",
        type=int,
        nargs=2,
        default=[16, 16],
        help="Minimum detection size [width height] for octave-based scaling",
    )
    parser.add_argument(
        "--max_ds",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Maximum detection size [width height] for octave-based scaling",
    )
    parser.add_argument(
        "--max_scale",
        type=float,
        default=None,
        help="Maximum scale for octave-based scaling",
    )

    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    detector = ACFDetector()
    detector.load(args.model)

    print(f"Loading image {args.image}...")
    image = load_image(args.image)

    print("Running detection...")
    print(f"Using {'fast' if args.use_fast_pyramid else 'original'} feature pyramid")

    scales = get_scales_octave_based(
        args.n_per_oct, args.n_oct_up, tuple(args.min_ds), args.max_scale
    )
    print(scales)
    vis_dir = None
    if args.feature_vis_dir:
        vis_dir = prepare_output_dir(args.feature_vis_dir)
        print(f"Feature visualizations will be saved to {vis_dir}")

    detections = detect_multiscale(
        detector=detector,
        image=image,
        window_size=args.max_ds,
        scales=scales,
        stride=args.stride,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        batch_size=args.batch_size,
        use_fast_pyramid=args.use_fast_pyramid,
        save_crops_dir=vis_dir,
    )

    print(f"Detected {len(detections)} faces")

    for i, det in enumerate(detections):
        x, y, w, h, score = det
        features = detector.extract_features(image, (x, y, w, h))
        feature_map = features.reshape(
            (detector.feature_resolution, detector.feature_resolution, 10)
        )

        print(f"  Face {i + 1}: x={x}, y={y}, w={w}, h={h}, score={score:.3f}")

        if vis_dir:
            vis_path = vis_dir / f"detection_{i + 1:03d}_score_{score:.3f}.jpg"

            figure = visualize_feature_map(
                feature_map, i + 1, score, image[y : y + h, x : x + w]
            )
            cv2.imwrite(str(vis_path), figure)
            print(f"    → Visualization saved to {vis_path}")

    if args.output or len(detections) > 0:
        vis_image = visualize_detections(image, detections)

        if args.output:
            vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(args.output, vis_bgr)
            print(f"Output saved to {args.output}")
        else:
            cv2.imshow("Detections", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
