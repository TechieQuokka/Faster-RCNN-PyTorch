"""
테스트용 더미 데이터셋 생성
"""

import os
import json
import random
from PIL import Image, ImageDraw
import numpy as np

def create_dummy_voc_dataset(output_dir, num_images=50):
    """
    더미 PASCAL VOC 형식 데이터셋 생성

    Args:
        output_dir: 출력 디렉토리
        num_images: 생성할 이미지 개수
    """
    # 디렉토리 생성
    voc_dir = os.path.join(output_dir, 'VOC2012')
    os.makedirs(os.path.join(voc_dir, 'JPEGImages'), exist_ok=True)
    os.makedirs(os.path.join(voc_dir, 'Annotations'), exist_ok=True)
    os.makedirs(os.path.join(voc_dir, 'ImageSets', 'Main'), exist_ok=True)

    # 클래스 이름
    classes = ['person', 'car', 'dog', 'cat', 'bicycle']

    # 이미지 ID 리스트
    train_ids = []
    val_ids = []

    print(f'더미 데이터셋 생성 중: {num_images}개 이미지')

    for i in range(num_images):
        img_id = f'{i:06d}'

        # 80% 학습, 20% 검증
        if i < int(num_images * 0.8):
            train_ids.append(img_id)
        else:
            val_ids.append(img_id)

        # 랜덤 이미지 생성 (600x400)
        width, height = 600, 400
        img = Image.new('RGB', (width, height), color=(
            random.randint(100, 200),
            random.randint(100, 200),
            random.randint(100, 200)
        ))
        draw = ImageDraw.Draw(img)

        # 랜덤 객체 그리기 (1-5개)
        num_objects = random.randint(1, 5)

        xml_content = f'''<annotation>
    <folder>VOC2012</folder>
    <filename>{img_id}.jpg</filename>
    <source>
        <database>Dummy VOC2012</database>
    </source>
    <size>
        <width>{width}</width>
        <height>{height}</height>
        <depth>3</depth>
    </size>
'''

        for _ in range(num_objects):
            # 랜덤 박스 생성
            x1 = random.randint(0, width - 100)
            y1 = random.randint(0, height - 100)
            box_w = random.randint(50, min(150, width - x1))
            box_h = random.randint(50, min(150, height - y1))
            x2 = x1 + box_w
            y2 = y1 + box_h

            # 박스 그리기
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 클래스 선택
            cls_name = random.choice(classes)

            # XML 어노테이션 추가
            xml_content += f'''    <object>
        <name>{cls_name}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{x1}</xmin>
            <ymin>{y1}</ymin>
            <xmax>{x2}</xmax>
            <ymax>{y2}</ymax>
        </bndbox>
    </object>
'''

        xml_content += '</annotation>'

        # 이미지 저장
        img_path = os.path.join(voc_dir, 'JPEGImages', f'{img_id}.jpg')
        img.save(img_path)

        # 어노테이션 저장
        ann_path = os.path.join(voc_dir, 'Annotations', f'{img_id}.xml')
        with open(ann_path, 'w') as f:
            f.write(xml_content)

    # ImageSets 파일 생성
    with open(os.path.join(voc_dir, 'ImageSets', 'Main', 'train.txt'), 'w') as f:
        f.write('\n'.join(train_ids))

    with open(os.path.join(voc_dir, 'ImageSets', 'Main', 'val.txt'), 'w') as f:
        f.write('\n'.join(val_ids))

    with open(os.path.join(voc_dir, 'ImageSets', 'Main', 'trainval.txt'), 'w') as f:
        f.write('\n'.join(train_ids + val_ids))

    print(f'완료! 학습: {len(train_ids)}개, 검증: {len(val_ids)}개')
    print(f'저장 위치: {voc_dir}')


if __name__ == '__main__':
    create_dummy_voc_dataset('data', num_images=50)
