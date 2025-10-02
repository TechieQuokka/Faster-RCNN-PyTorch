"""
YOLO format PASCAL VOC Dataset for Faster R-CNN
Converts YOLO format (normalized center coordinates) to Faster R-CNN format (corner coordinates)
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms


class YOLOVOCDataset(Dataset):
    """
    PASCAL VOC Dataset in YOLO format

    YOLO format: class_id x_center y_center width height (all normalized 0-1)
    Converts to: boxes (x1, y1, x2, y2) in pixel coordinates, labels (class_id + 1 for background)
    """

    def __init__(self, csv_file, img_dir, label_dir, transform=None, resize_to=(600, 600)):
        """
        Args:
            csv_file: Path to CSV file with image,label pairs
            img_dir: Directory with images
            label_dir: Directory with YOLO format labels
            transform: Image transformations
            resize_to: Target size (height, width) for resizing images
        """
        self.annotations = pd.read_csv(csv_file, header=None, names=['image', 'label'])
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.resize_to = resize_to

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

        # Resize image
        if self.resize_to is not None:
            target_height, target_width = self.resize_to
            image = image.resize((target_width, target_height))
            scale_x = target_width / orig_width
            scale_y = target_height / orig_height
        else:
            scale_x = 1.0
            scale_y = 1.0
            target_width, target_height = orig_width, orig_height

        # Load labels
        label_name = self.annotations.iloc[idx, 1]
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        labels = []

        # Read YOLO format labels
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

                # Convert YOLO format (normalized center) to corner format (pixel coordinates in original size)
                x1 = (x_center - width / 2) * orig_width
                y1 = (y_center - height / 2) * orig_height
                x2 = (x_center + width / 2) * orig_width
                y2 = (y_center + height / 2) * orig_height

                # Scale to resized image
                x1 = x1 * scale_x
                y1 = y1 * scale_y
                x2 = x2 * scale_x
                y2 = y2 * scale_y

                # Clip to image boundaries
                x1 = max(0, min(x1, target_width))
                y1 = max(0, min(y1, target_height))
                x2 = max(0, min(x2, target_width))
                y2 = max(0, min(y2, target_height))

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

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = transforms.ToTensor()(image)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        return image, target

    def get_class_name(self, class_id):
        """Get class name from class ID (0-indexed)"""
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return 'background'


def get_yolo_voc_datasets(root_dir, train_csv='train.csv', test_csv='test.csv',
                          img_dir='images', label_dir='labels'):
    """
    Create train and test datasets from YOLO format PASCAL VOC

    Args:
        root_dir: Root directory containing all data
        train_csv: Training CSV filename
        test_csv: Test CSV filename
        img_dir: Images directory name
        label_dir: Labels directory name

    Returns:
        train_dataset, test_dataset
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = YOLOVOCDataset(
        csv_file=os.path.join(root_dir, train_csv),
        img_dir=os.path.join(root_dir, img_dir),
        label_dir=os.path.join(root_dir, label_dir),
        transform=transform
    )

    test_dataset = YOLOVOCDataset(
        csv_file=os.path.join(root_dir, test_csv),
        img_dir=os.path.join(root_dir, img_dir),
        label_dir=os.path.join(root_dir, label_dir),
        transform=transform
    )

    return train_dataset, test_dataset
