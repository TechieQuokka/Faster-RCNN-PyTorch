"""
Train Faster R-CNN on PASCAL VOC Dataset (YOLO format) with Config Support
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import argparse
from tqdm import tqdm

from faster_rcnn.models.faster_rcnn import build_faster_rcnn
from faster_rcnn.yolo_voc_dataset import get_yolo_voc_datasets
from faster_rcnn.data.collate import collate_fn
from faster_rcnn.engine.evaluator import VOCEvaluator
from faster_rcnn.utils.config import load_config


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_rpn_cls_loss = 0
    total_rpn_box_loss = 0
    total_rcnn_cls_loss = 0
    total_rcnn_box_loss = 0

    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(dataloader):
        # Move to device
        images = [img.to(device) for img in images]
        # Move only tensor values to device (skip tuples like original_size, resized_size)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses
        total_loss += losses.item()
        total_rpn_cls_loss += loss_dict['loss_rpn_cls'].item()
        total_rpn_box_loss += loss_dict['loss_rpn_reg'].item()
        total_rcnn_cls_loss += loss_dict['loss_det_cls'].item()
        total_rcnn_box_loss += loss_dict['loss_det_reg'].item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f'Epoch [{epoch}], Batch [{batch_idx + 1}/{len(dataloader)}], '
                  f'Loss: {avg_loss:.4f}, '
                  f'RPN Cls: {total_rpn_cls_loss / (batch_idx + 1):.4f}, '
                  f'RPN Box: {total_rpn_box_loss / (batch_idx + 1):.4f}, '
                  f'RCNN Cls: {total_rcnn_cls_loss / (batch_idx + 1):.4f}, '
                  f'RCNN Box: {total_rcnn_box_loss / (batch_idx + 1):.4f}')

    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(dataloader)

    print(f'\nEpoch [{epoch}] Summary:')
    print(f'Time: {epoch_time:.2f}s')
    print(f'Avg Loss: {avg_loss:.4f}')
    print(f'RPN Cls Loss: {total_rpn_cls_loss / len(dataloader):.4f}')
    print(f'RPN Box Loss: {total_rpn_box_loss / len(dataloader):.4f}')
    print(f'RCNN Cls Loss: {total_rcnn_cls_loss / len(dataloader):.4f}')
    print(f'RCNN Box Loss: {total_rcnn_box_loss / len(dataloader):.4f}\n')

    return avg_loss


def validate(model, dataloader, device):
    """Validate the model"""
    # Keep model in train mode for loss calculation, but disable gradients
    model.train()
    total_loss = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            # Move only tensor values to device (skip tuples like original_size, resized_size)
            targets = [{k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in t.items()} for t in targets]

            # Forward pass (train mode returns loss_dict)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Validation Loss: {avg_loss:.4f}\n')

    return avg_loss


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Faster R-CNN on YOLO VOC Dataset')
    parser.add_argument('--config', type=str, default='faster_rcnn/configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    args = parser.parse_args()

    # Load config
    print(f'Loading config from {args.config}...')
    config = load_config(args.config)

    # Override config with command line arguments
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr

    # Paths
    CHECKPOINT_DIR = 'checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n')

    # Load datasets
    print('Loading PASCAL VOC dataset...')
    train_dataset, val_dataset = get_yolo_voc_datasets(
        root_dir=config.dataset.root,
        train_csv=config.dataset.train_csv,
        test_csv=config.dataset.test_csv,
        img_dir=config.dataset.img_dir,
        label_dir=config.dataset.label_dir,
        config=config
    )
    print(f'Train samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}\n')

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Create model
    print('Creating Faster R-CNN model...')
    model = build_faster_rcnn(
        num_classes=config.model.num_classes,
        backbone_name=config.model.backbone,
        pretrained_backbone=config.model.pretrained_backbone,
        # RPN 파라미터
        rpn_anchor_sizes=tuple(config.rpn.anchor_sizes),
        rpn_anchor_ratios=tuple(config.rpn.anchor_ratios),
        rpn_nms_thresh=config.rpn.nms_thresh,
        rpn_pre_nms_top_n_train=config.rpn.pre_nms_top_n_train,
        rpn_post_nms_top_n_train=config.rpn.post_nms_top_n_train,
        # RoI Head 파라미터
        roi_output_size=config.roi_head.roi_output_size,
        roi_fg_iou_thresh=config.roi_head.fg_iou_thresh,
        roi_bg_iou_thresh=config.roi_head.bg_iou_thresh,
        roi_batch_size_per_image=config.roi_head.batch_size_per_image,
        roi_positive_fraction=config.roi_head.positive_fraction,
        roi_score_thresh=config.roi_head.score_thresh,
        roi_nms_thresh=config.roi_head.nms_thresh,
        roi_detection_per_img=config.roi_head.detection_per_img,
    )
    model = model.to(device)
    print(f'Model created with {config.model.num_classes} classes\n')

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training.get('lr_step_size', 3),
        gamma=config.training.get('lr_gamma', 0.1)
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resumed from epoch {checkpoint["epoch"]}\n')

    # Create evaluator if specified in config
    evaluator = None
    if hasattr(config, 'evaluation'):
        if config.evaluation.metric == 'voc':
            evaluator = VOCEvaluator(iou_threshold=config.evaluation.get('iou_threshold', 0.5))
            print(f'Using VOC mAP@{config.evaluation.get("iou_threshold", 0.5)} evaluation\n')
        elif config.evaluation.metric == 'coco':
            from faster_rcnn.engine.evaluator import COCOEvaluator
            evaluator = COCOEvaluator()
            print('Using COCO mAP evaluation\n')

    # Training loop
    print('Starting training...\n')
    best_val_loss = float('inf')
    best_map = 0.0

    for epoch in range(start_epoch, config.training.num_epochs + 1):
        print(f'Epoch {epoch}/{config.training.num_epochs}')
        print('-' * 50)

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Evaluate if evaluator is available
        if evaluator is not None:
            print('Running evaluation...')
            metrics = evaluator.evaluate(model, val_loader, device)
            if 'mAP' in metrics:
                print(f'mAP: {metrics["mAP"]:.4f}')
                current_map = metrics['mAP']
            else:
                current_map = 0.0
        else:
            current_map = 0.0

        # Step scheduler
        scheduler.step()

        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config,
        }, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')

        # Save best model (based on mAP if available, otherwise val_loss)
        if evaluator is not None:
            if current_map > best_map:
                best_map = current_map
                best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'mAP': current_map,
                    'config': config,
                }, best_model_path)
                print(f'Best model saved: {best_model_path} (mAP: {best_map:.4f})')
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': config,
                }, best_model_path)
                print(f'Best model saved: {best_model_path} (Val Loss: {best_val_loss:.4f})')

        print()

    print('Training completed!')
    if evaluator is not None:
        print(f'Best mAP: {best_map:.4f}')
    else:
        print(f'Best validation loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    main()
