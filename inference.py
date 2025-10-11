#!/usr/bin/env python
import argparse
import cv2
from acf.model import ACFDetector
from acf.inference import detect_multiscale, visualize_detections
from acf.preprocessing import load_image


def main():
    parser = argparse.ArgumentParser(description='Run ACF face detection on an image')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output image (optional)')
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
                       help='Batch size for inference (default: 32)')
    
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
    
    for i, det in enumerate(detections):
        x, y, w, h, score = det
        print(f"  Face {i+1}: x={x}, y={y}, w={w}, h={h}, score={score:.3f}")
    
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
