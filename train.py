#!/usr/bin/env python
import argparse
import os
from acf.model import ACFDetector, MemoryEfficientBootstrapWithHeap
from acf.preprocessing import (
    AnnotationSetting,
    get_muct_annotations,
    get_muct_training_data,
    get_widerface_training_data,
    parse_wider_face_annotation,
)


from dataclasses import dataclass
from typing import Tuple, List, Optional, Literal


@dataclass
class Args:
    annotation_file: str
    image_dir: str
    output_model: Optional[str]

    window_size: Tuple[int, int]
    hidden_sizes: List[int]
    feature_resolution: int

    learning_rate: float
    batch_size: int
    epochs: int
    max_images: Optional[int]

    val_annotation_file: Optional[str]
    val_image_dir: Optional[str]
    max_val_images: Optional[int]

    patience: int
    selection_metric: Literal["precision", "f1", "f_beta", "val_loss", "map"]

    pos_iou_thresh: float
    neg_iou_thresh: float
    hard_neg_iou_range: Tuple[float, float]
    num_neg_per_pos: int

    bootstrap_rounds: int
    num_mining_images: int

    acceptable_blur: Optional[List[int]]
    acceptable_expression: Optional[List[int]]
    acceptable_illumination: Optional[List[int]]
    acceptable_occlusion: Optional[List[int]]
    acceptable_pose: Optional[List[int]]

    filter_invalid: bool
    checkpoint: Optional[str]

    model: Literal["mlp", "cnn", "ada", "gbm"]
    dataset: Literal["widerface", "muct"]


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Train ACF face detector")

    parser.add_argument(
        "--annotation_file",
        type=str,
        default="data/wider_face_split/wider_face_train_bbx_gt.txt",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/WIDER_train/images/",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--window_size",
        type=int,
        nargs=2,
        default=[64, 64],
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[512, 256],
    )
    parser.add_argument(
        "--feature_resolution",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--val_annotation_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--val_image_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max_val_images",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="f_beta",
        choices=["precision", "f1", "f_beta", "val_loss", "map"],
    )

    parser.add_argument(
        "--pos_iou_thresh",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--neg_iou_thresh",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--hard_neg_iou_range",
        type=float,
        nargs=2,
        default=(0.1, 0.3),
    )
    parser.add_argument(
        "--num_neg_per_pos",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--bootstrap_rounds",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num_mining_images",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--acceptable_blur",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--acceptable_expression",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--acceptable_illumination",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--acceptable_occlusion",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--acceptable_pose",
        type=int,
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--filter_invalid",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "cnn", "ada", "gbm"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="widerface",
        choices=["widerface", "muct"],
    )

    args = parser.parse_args()

    # Normalize container types
    args.window_size = tuple(args.window_size)
    args.hard_neg_iou_range = tuple(args.hard_neg_iou_range)

    return Args(**vars(args))


def main():
    args = parse_args()

    ww, wh = args.window_size
    fr = args.feature_resolution
    lr = args.learning_rate
    e = args.epochs

    if args.output_model is None:
        hidden_str = "_".join(map(str, args.hidden_sizes))
        args.output_model = f"models/acf_w{ww}x{wh}_h{hidden_str}_r{fr}_lr{lr}_e{e}.pkl"

    if args.model == "gbm" or args.model == "ada":
        args.epochs = 1

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

    print("Initializing ACF detector...")
    print("Configuration:")
    print(f"  - Model: {args.model}")
    print(f"  - Window size: {args.window_size}")
    print(f"  - Hidden layers: {args.hidden_sizes}")
    print(
        f"  - Feature resolution: {args.feature_resolution}x{args.feature_resolution}"
    )
    print(
        f"  - Feature vector size: {10 * args.feature_resolution * args.feature_resolution}"
    )
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Selection Metric: {args.selection_metric}")
    print(f"  - Positive IoU: {args.pos_iou_thresh}")
    print(f"  - Negative IoU: {args.neg_iou_thresh}")
    print(f"  - Hard Negative IoU Range: {args.hard_neg_iou_range}")
    print(f"  - Number of negatives per positive: {args.num_neg_per_pos}")
    print(f"  - Acceptable blur: {args.acceptable_blur}")
    print(f"  - Acceptable expression: {args.acceptable_expression}")
    print(f"  - Acceptable illumination: {args.acceptable_illumination}")
    print(f"  - Acceptable occlusion: {args.acceptable_occlusion}")
    print(f"  - Acceptable pose: {args.acceptable_pose}")
    print(f"  - Filter invalid: {args.filter_invalid}")

    detector = ACFDetector(
        window_size=tuple(args.window_size),
        hidden_sizes=args.hidden_sizes,
        feature_resolution=args.feature_resolution,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        selection_metric=args.selection_metric,
        pos_iou_thresh=args.pos_iou_thresh,
        neg_iou_thresh=args.neg_iou_thresh,
        hard_neg_iou_range=tuple(args.hard_neg_iou_range),
        num_neg_per_pos=args.num_neg_per_pos,
        model=args.model,
    )

    if args.checkpoint is not None:
        print(f"Loading checkpoint {args.checkpoint}")
        detector.load(args.checkpoint)

    print("\nStarting training...")

    if args.dataset == "widerface":
        annotation_setting = AnnotationSetting(
            acceptable_blur=args.acceptable_blur,
            acceptable_expression=args.acceptable_expression,
            acceptable_illumination=args.acceptable_illumination,
            acceptable_occlusion=args.acceptable_occlusion,
            acceptable_pose=args.acceptable_pose,
            filter_invalid=args.filter_invalid,
        )

        assert args.val_annotation_file
        assert args.val_image_dir
        X_train, y_train, X_val, y_val = get_widerface_training_data(
            annotation_file=args.annotation_file,
            val_annotation_file=args.val_annotation_file,
            image_dir=args.image_dir,
            val_image_dir=args.val_image_dir,
            annotation_setting=annotation_setting,
            feature_resolution=args.feature_resolution,
            hard_neg_iou_range=args.hard_neg_iou_range,
            max_images=args.max_images,
            max_val_images=args.max_val_images,
            neg_iou_thresh=args.neg_iou_thresh,
            num_neg_per_pos=args.num_neg_per_pos,
            pos_iou_thresh=args.pos_iou_thresh,
            window_size=args.window_size,
        )

        annotations = parse_wider_face_annotation(
            args.annotation_file, AnnotationSetting(True)
        )
    elif args.dataset == "muct":
        annotations, _ = get_muct_annotations(args.annotation_file, args.image_dir)
        X_train, y_train, X_val, y_val = get_muct_training_data(
            annotation_file=args.annotation_file,
            image_dir=args.image_dir,
            feature_resolution=args.feature_resolution,
            hard_neg_iou_range=args.hard_neg_iou_range,
            max_images=args.max_images,
            neg_iou_thresh=args.neg_iou_thresh,
            num_neg_per_pos=args.num_neg_per_pos,
            pos_iou_thresh=args.pos_iou_thresh,
            window_size=args.window_size,
        )

    # X_train, y_train, X_val, y_val = detector.get_train_data(
    #     dataset_type=args.dataset,
    #     annotation_setting=annotation_setting,
    #     annotation_file=args.annotation_file,
    #     image_base_dir=args.image_dir,
    #     max_images=args.max_images,
    #     val_annotation_file=args.val_annotation_file,
    #     val_image_base_dir=args.val_image_dir,
    #     max_val_images=args.max_val_images,
    # )

    bootstrap = MemoryEfficientBootstrapWithHeap(
        detector=detector,
        dataset=args.dataset,
        annotations=annotations,
    )
    bootstrap.train_with_bootstrap(
        X_train,
        y_train,
        X_val,
        y_val,
        early_stopping_patience=args.patience,
        mining_annotation_file=args.annotation_file,
        mining_image_base_dir=args.image_dir,
        num_mining_images=args.num_mining_images,
        bootstrap_rounds=args.bootstrap_rounds,
    )

    bootstrap.detector.save(args.output_model)
    print(f"\nTraining complete! Model saved to {args.output_model}")


if __name__ == "__main__":
    main()
