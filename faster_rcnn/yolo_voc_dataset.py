"""
YOLO format PASCAL VOC Dataset for Faster R-CNN
Converts YOLO format (normalized center coordinates) to Faster R-CNN format (corner coordinates)
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class YOLOVOCDataset(Dataset):
    """
    PASCAL VOC Dataset in YOLO format

    YOLO format: class_id x_center y_center width height (all normalized 0-1)
    Converts to: boxes (x1, y1, x2, y2) in pixel coordinates, labels (class_id + 1 for background)
    """

    def __init__(self, csv_file, img_dir, label_dir, transform=None):
        """
        Args:
            csv_file: Path to CSV file with image,label pairs
            img_dir: Directory with images
            label_dir: Directory with YOLO format labels
            transform: Transform pipeline (includes resize, augmentation, normalization)
        """
        self.annotations = pd.read_csv(csv_file, header=None, names=['image', 'label'])
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        # PASCAL VOC 20 classes
        self.classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load image
        img_name = self.annotations.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Get original image size
        orig_width, orig_height = image.size

        # Load labels
        label_name = self.annotations.iloc[idx, 1]
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        labels = []

        # Read YOLO format labels
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Convert YOLO format (normalized center) to corner format (pixel coordinates)
                    x1 = (x_center - width / 2) * orig_width
                    y1 = (y_center - height / 2) * orig_height
                    x2 = (x_center + width / 2) * orig_width
                    y2 = (y_center + height / 2) * orig_height

                    # Clip to image boundaries
                    x1 = max(0, min(x1, orig_width))
                    y1 = max(0, min(y1, orig_height))
                    x2 = max(0, min(x2, orig_width))
                    y2 = max(0, min(y2, orig_height))

                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue

                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id + 1)  # +1 for background class at index 0

        # Convert to tensors
        if len(boxes) == 0:
            # Handle images with no valid boxes
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # Prepare target
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        # Apply transformations (resize, augmentation, normalization)
        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def get_class_name(self, class_id):
        """Get class name from class ID (0-indexed)"""
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return 'background'


def get_yolo_voc_datasets(config=None, root_dir=None, train_csv='train.csv',
                          test_csv='test.csv', img_dir='images', label_dir='labels'):
    """
    Create train and test datasets from YOLO format PASCAL VOC

    Args:
        config: Configuration object (preferred)
        root_dir: Root directory containing all data (fallback)
        train_csv: Training CSV filename
        test_csv: Test CSV filename
        img_dir: Images directory name
        label_dir: Labels directory name

    Returns:
        train_dataset, test_dataset
    """
    # Import here to avoid circular dependency
    from .data.transforms import get_transform

    # Get parameters from config
    if config is not None:
        root_dir = config.dataset.root
        train_csv = config.dataset.train_csv
        test_csv = config.dataset.test_csv
        img_dir = config.dataset.img_dir
        label_dir = config.dataset.label_dir

    # Define transforms (resize, augmentation, normalization)
    train_transform = get_transform(train=True, config=config)
    val_transform = get_transform(train=False, config=config)

    train_dataset = YOLOVOCDataset(
        csv_file=os.path.join(root_dir, train_csv),
        img_dir=os.path.join(root_dir, img_dir),
        label_dir=os.path.join(root_dir, label_dir),
        transform=train_transform
    )

    test_dataset = YOLOVOCDataset(
        csv_file=os.path.join(root_dir, test_csv),
        img_dir=os.path.join(root_dir, img_dir),
        label_dir=os.path.join(root_dir, label_dir),
        transform=val_transform
    )

    return train_dataset, test_dataset
