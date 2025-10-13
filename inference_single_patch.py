#!/usr/bin/env python
import argparse
from acf.model import ACFDetector
from acf.preprocessing import load_image
import torch


def main():
    parser = argparse.ArgumentParser(description='Run ACF face detection on an image')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--nms_threshold', type=float, default=0.3,
                       help='NMS IoU threshold')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    detector = ACFDetector()
    detector.load(args.model)
    
    print(f"Loading image {args.image}...")
    image = load_image(args.image)
    
    print("Extractng feature...")
    W, H, _ = image.shape
    feature = detector.extract_features(image, (0, 0, W, H))
    feature_tensor = (
        torch.from_numpy(feature).float().to(detector.device)
    )
    feature_tensor = feature_tensor.unsqueeze(0)

    print("Running inference...")
    detector.classifier.eval()
    with torch.no_grad():
        output = detector.classifier(feature_tensor)
        prob = torch.softmax(output, dim=1)
        score = prob[:, 1].cpu().numpy()

    print(f"Model output: {output}")
    print(f"Model prob: {prob}")
    print(f"Model score: {score}")



if __name__ == '__main__':
    main()
