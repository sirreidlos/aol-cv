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
    channels_names = ['L', 'U', 'V', 'M', 'HOG1', 'HOG2', 'HOG3', 'HOG4', 'HOG5', 'HOG6']
    
    fig_height = 2 * 16 * 2  
    fig_width = 5 * 16 * 2   
    figure = np.ones((fig_height + 60, fig_width + 20), dtype=np.uint8) * 255
    
    title = f"Detection {detection_idx} (score: {score:.3f})"
    cv2.putText(figure, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 0, 0), 1)
    
    for idx, (channel, name) in enumerate(zip(feature_map.transpose(2, 0, 1), channels_names)):
        row = idx // 5
        col = idx % 5
        
        y_start = 60 + row * 16 * 2
        x_start = 10 + col * 16 * 2
        
        normalized = normalize_channel(channel)
        upscaled = cv2.resize(normalized, (32, 32), interpolation=cv2.INTER_NEAREST)
        
        figure[y_start:y_start+32, x_start:x_start+32] = upscaled
        
        cv2.putText(figure, name, (x_start, y_start - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    output_path = output_dir / f"detection_{detection_idx:03d}_score_{score:.3f}.jpg"
    cv2.imwrite(str(output_path), figure)
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Run ACF face detection with feature visualization')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output detection image (optional)')
    parser.add_argument('--feature_vis_dir', type=str, default=None,
                       help='Directory to save feature channel visualizations')
    parser.add_argument('--scales', type=float, nargs='+', 
                       default=[0.5, 0.75, 1.0, 1.25, 1.5],
                       help='Scales for multi-scale detection')
    parser.add_argument('--stride', type=int, default=8,
                       help='Sliding window stride')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                       help='Minimum confidence score')
    parser.add_argument('--nms_threshold', type=float, default=0.3,
                       help='NMS IoU threshold')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    detector = ACFDetector()
    detector.load(args.model)
    
    print(f"Loading image {args.image}...")
    image = load_image(args.image)
    
    print("Running detection...")
    detections = detect_multiscale(
        detector=detector,
        image=image,
        scales=args.scales,
        stride=args.stride,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        batch_size=args.batch_size
    )
    
    print(f"Detected {len(detections)} faces")
    
    vis_dir = None
    if args.feature_vis_dir:
        vis_dir = prepare_output_dir(args.feature_vis_dir)
        print(f"Feature visualizations will be saved to {vis_dir}")
    
    for i, det in enumerate(detections):
        x, y, w, h, score = det
        features = detector.extract_features(image, (x, y, w, h))
        feature_map = features.reshape((detector.feature_resolution, detector.feature_resolution, 10))
        
        print(f"  Face {i+1}: x={x}, y={y}, w={w}, h={h}, score={score:.3f}")
        
        if vis_dir:
            vis_path = visualize_feature_map(feature_map, i+1, vis_dir, score)
            print(f"    → Visualization saved to {vis_path}")
    
    if args.output or len(detections) > 0:
        vis_image = visualize_detections(image, detections)
        
        if args.output:
            vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(args.output, vis_bgr)
            print(f"Output saved to {args.output}")
        else:
            cv2.imshow('Detections', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
