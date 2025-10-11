#!/usr/bin/env python
import argparse
import os
from acf.model import ACFDetector


def main():
    parser = argparse.ArgumentParser(description='Train ACF face detector')
    parser.add_argument('--annotation_file', type=str, default="data/wider_face_split/wider_face_train_bbx_gt.txt",
                       help='Path to WIDER FACE annotation file')
    parser.add_argument('--image_dir', type=str, default="data/WIDER_train/images/",
                       help='Path to image directory')
    parser.add_argument('--output_model', type=str, default='models/acf_detector.pkl',
                       help='Path to save trained model')
    parser.add_argument('--window_size', type=int, nargs=2, default=[64, 64],
                       help='Detection window size (width height)')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[512, 256],
                       help='Hidden layer sizes for MLP (default: 512 256)')
    parser.add_argument('--feature_resolution', type=int, default=16,
                       help='Feature resolution for each channel (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of training images (None for all)')
    parser.add_argument('--val_annotation_file', type=str, default=None,
                       help='Path to validation WIDER FACE annotation file (optional)')
    parser.add_argument('--val_image_dir', type=str, default=None,
                       help='Path to validation image directory (optional)')
    parser.add_argument('--max_val_images', type=int, default=None,
                       help='Maximum number of validation images (None for all)')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='Number of epochs without improvement before early stopping (default: 3)')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    
    print("Initializing ACF detector...")
    print("Configuration:")
    print(f"  - Window size: {args.window_size}")
    print(f"  - Hidden layers: {args.hidden_sizes}")
    print(f"  - Feature resolution: {args.feature_resolution}x{args.feature_resolution}")
    print(f"  - Feature vector size: {10 * args.feature_resolution * args.feature_resolution}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    
    detector = ACFDetector(
        window_size=tuple(args.window_size),
        hidden_sizes=args.hidden_sizes,
        feature_resolution=args.feature_resolution,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    print("\nStarting training...")
    X_train, y_train, X_val, y_val = detector.get_train_data(
        annotation_file=args.annotation_file,
        image_base_dir=args.image_dir,
        max_images=args.max_images,
        val_annotation_file=args.val_annotation_file,
        val_image_base_dir=args.val_image_dir,
        max_val_images=args.max_val_images,
    )
    detector.train(X_train, y_train, X_val, y_val,
        early_stopping_patience=args.early_stopping_patience
    )
    
    detector.save(args.output_model)
    print(f"\nTraining complete! Model saved to {args.output_model}")


if __name__ == '__main__':
    main()
