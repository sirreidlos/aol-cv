#!/usr/bin/env python
import argparse
import cv2
import numpy as np
import shutil
from pathlib import Path
from acf.model import ACFDetector
from acf.inference import detect_multiscale, visualize_detections
from acf.preprocessing import load_image


def prepare_output_dir(output_dir):
    """Create output directory, clearing it if it already exists."""
    path = Path(output_dir)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_channel(channel):
    """Normalize a channel to 0-255 for visualization."""
    ch_min = channel.min()
    ch_max = channel.max()

    if ch_max - ch_min < 1e-6:
        return np.zeros_like(channel, dtype=np.uint8)

    normalized = (channel - ch_min) / (ch_max - ch_min) * 255
    return normalized.astype(np.uint8)


def visualize_feature_map(feature_map, detection_idx, output_dir, score):
    """
    Visualize all 10 feature channels as a grid.
    feature_map: [16, 16, 10]
    """
    channels_names = [
        "L",
        "U",
        "V",
        "M",
        "HOG1",
        "HOG2",
        "HOG3",
        "HOG4",
        "HOG5",
        "HOG6",
    ]

    fig_height = 2 * 16 * 2
    fig_width = 5 * 16 * 2
    figure = np.ones((fig_height + 60, fig_width + 20), dtype=np.uint8) * 255

    title = f"Detection {detection_idx} (score: {score:.3f})"
    cv2.putText(figure, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    for idx, (channel, name) in enumerate(
        zip(feature_map.transpose(2, 0, 1), channels_names)
    ):
        row = idx // 5
        col = idx % 5

        y_start = 60 + row * 16 * 2
        x_start = 10 + col * 16 * 2

        normalized = normalize_channel(channel)
        upscaled = cv2.resize(normalized, (32, 32), interpolation=cv2.INTER_NEAREST)

        figure[y_start : y_start + 32, x_start : x_start + 32] = upscaled

        cv2.putText(
            figure,
            name,
            (x_start, y_start - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )

    output_path = output_dir / f"detection_{detection_idx:03d}_score_{score:.3f}.jpg"
    cv2.imwrite(str(output_path), figure)
    return output_path


def get_scales_octave_based(n_per_oct=8, n_oct_up=0, min_ds=(16, 16), max_scale=None):
    scales = []
    scale_factor = 2 ** (-1.0 / n_per_oct)

    current_scale = 2**n_oct_up

    while True:
        scales.append(current_scale)
        if max_scale and current_scale < max_scale:
            break
        current_scale *= scale_factor

        if (
            current_scale * min_ds[0] < min_ds[0]
            or current_scale * min_ds[1] < min_ds[1]
        ):
            break

    return scales


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
        default=32,
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

    detections = detect_multiscale(
        detector=detector,
        image=image,
        scales=scales,
        stride=args.stride,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        batch_size=args.batch_size,
        use_fast_pyramid=args.use_fast_pyramid,
    )

    print(f"Detected {len(detections)} faces")

    vis_dir = None
    if args.feature_vis_dir:
        vis_dir = prepare_output_dir(args.feature_vis_dir)
        print(f"Feature visualizations will be saved to {vis_dir}")

    for i, det in enumerate(detections):
        x, y, w, h, score = det
        features = detector.extract_features(image, (x, y, w, h))
        feature_map = features.reshape(
            (detector.feature_resolution, detector.feature_resolution, 10)
        )

        print(f"  Face {i + 1}: x={x}, y={y}, w={w}, h={h}, score={score:.3f}")

        if vis_dir:
            vis_path = visualize_feature_map(feature_map, i + 1, vis_dir, score)
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
