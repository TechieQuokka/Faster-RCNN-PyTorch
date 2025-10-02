"""
Train Faster R-CNN on PASCAL VOC Dataset (YOLO format)
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

from faster_rcnn.models.faster_rcnn import FasterRCNN
from faster_rcnn.yolo_voc_dataset import get_yolo_voc_datasets
from faster_rcnn.data.collate import collate_fn


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
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
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
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Validation Loss: {avg_loss:.4f}\n')

    return avg_loss


def main():
    # Hyperparameters
    NUM_CLASSES = 21  # 20 PASCAL VOC classes + background
    BATCH_SIZE = 4
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    NUM_WORKERS = 4

    # Paths
    DATA_ROOT = '/home/beethoven/workspace/deeplearning/deeplearning-project/Faster_R-CNN'
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
        root_dir=DATA_ROOT,
        train_csv='train.csv',
        test_csv='test.csv',
        img_dir='images',
        label_dir='labels'
    )
    print(f'Train samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}\n')

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Create model
    print('Creating Faster R-CNN model...')
    model = FasterRCNN(num_classes=NUM_CLASSES, backbone_name='resnet50')
    model = model.to(device)
    print(f'Model created with {NUM_CLASSES} classes\n')

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    print('Starting training...\n')
    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'Epoch {epoch}/{NUM_EPOCHS}')
        print('-' * 50)

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'faster_rcnn_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(CHECKPOINT_DIR, 'faster_rcnn_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)
            print(f'Best model saved: {best_model_path}')

        print()

    print('Training completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    main()
