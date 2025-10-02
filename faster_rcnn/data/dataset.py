"""
데이터셋 클래스
- COCO 형식 데이터셋
- PASCAL VOC 데이터셋
- Custom 데이터셋
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import xml.etree.ElementTree as ET


class COCODataset(Dataset):
    """
    COCO 형식 데이터셋

    Args:
        root: 이미지 루트 디렉토리
        annFile: 어노테이션 JSON 파일 경로
        transforms: 데이터 변환 함수 (옵션)
    """

    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.transforms = transforms

        # COCO 어노테이션 로드
        with open(annFile, 'r') as f:
            self.coco_data = json.load(f)

        # 이미지 ID와 어노테이션 매핑
        self.image_ids = [img['id'] for img in self.coco_data['images']]
        self.image_info = {img['id']: img for img in self.coco_data['images']}

        # 카테고리 ID 매핑 (1부터 시작하도록)
        self.cat_ids = {cat['id']: idx + 1 for idx, cat in enumerate(self.coco_data['categories'])}
        self.cat_names = {idx + 1: cat['name'] for idx, cat in enumerate(self.coco_data['categories'])}

        # 이미지별 어노테이션 그룹화
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.image_info[img_id]

        # 이미지 로드
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        # 어노테이션 로드
        anns = self.img_to_anns.get(img_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            # COCO bbox: [x, y, width, height]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_ids[ann['category_id']])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        # Tensor로 변환
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd
        }

        # 변환 적용
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def get_class_names(self):
        """클래스 이름 반환"""
        return self.cat_names


class VOCDataset(Dataset):
    """
    PASCAL VOC 형식 데이터셋

    Args:
        root: VOC 데이터셋 루트 디렉토리
        year: 년도 ('2007' 또는 '2012')
        image_set: 'train', 'val', 'trainval', 'test'
        transforms: 데이터 변환 함수 (옵션)
    """

    def __init__(self, root, year='2012', image_set='train', transforms=None):
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transforms = transforms

        # 클래스 이름 (배경 제외)
        self.classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(self.classes)}

        # 이미지 리스트 로드
        voc_root = os.path.join(root, f'VOC{year}')
        image_set_file = os.path.join(voc_root, 'ImageSets', 'Main', f'{image_set}.txt')

        with open(image_set_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        self.image_dir = os.path.join(voc_root, 'JPEGImages')
        self.annotation_dir = os.path.join(voc_root, 'Annotations')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # 이미지 로드
        img_path = os.path.join(self.image_dir, f'{img_id}.jpg')
        img = Image.open(img_path).convert('RGB')

        # 어노테이션 로드 (XML)
        ann_path = os.path.join(self.annotation_dir, f'{img_id}.xml')
        boxes, labels = self.parse_voc_xml(ann_path)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        # 변환 적용
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def parse_voc_xml(self, xml_path):
        """VOC XML 어노테이션 파싱"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            # 클래스 이름
            name = obj.find('name').text
            if name not in self.class_to_idx:
                continue

            label = self.class_to_idx[name]

            # Bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return boxes, labels

    def get_class_names(self):
        """클래스 이름 반환"""
        return {idx + 1: cls for idx, cls in enumerate(self.classes)}
