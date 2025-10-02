"""
학습 루프
- Epoch 기반 학습
- 손실 계산 및 최적화
- 체크포인트 저장
- 학습률 스케줄링
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


class Trainer:
    """
    Faster R-CNN 학습 클래스

    Args:
        model: Faster R-CNN 모델
        optimizer: 옵티마이저
        device: 학습 디바이스
        checkpoint_dir: 체크포인트 저장 디렉토리
        print_freq: 로그 출력 빈도
    """

    def __init__(
        self,
        model,
        optimizer,
        device='cuda',
        checkpoint_dir='checkpoints',
        print_freq=10
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.print_freq = print_freq

        # 체크포인트 디렉토리 생성
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 모델을 디바이스로 이동
        self.model.to(device)

    def train_one_epoch(self, data_loader, epoch, scheduler=None):
        """
        한 Epoch 학습

        Args:
            data_loader: 학습 데이터 로더
            epoch: 현재 Epoch
            scheduler: 학습률 스케줄러 (옵션)

        Returns:
            avg_loss: 평균 손실
        """
        self.model.train()

        epoch_loss = 0.0
        loss_dict_reduced = {}

        # Progress bar
        pbar = tqdm(data_loader, desc=f'Epoch {epoch}')

        for iteration, (images, targets) in enumerate(pbar):
            # 데이터를 디바이스로 이동
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Forward pass
            losses = self.model(images, targets)

            # 전체 손실
            loss = losses['loss_total']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 손실 기록
            epoch_loss += loss.item()

            # 손실 딕셔너리 누적
            for k, v in losses.items():
                if k not in loss_dict_reduced:
                    loss_dict_reduced[k] = 0.0
                loss_dict_reduced[k] += v.item()

            # Progress bar 업데이트
            if iteration % self.print_freq == 0:
                avg_loss = epoch_loss / (iteration + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })

            # 학습률 스케줄러 (step마다 업데이트하는 경우)
            if scheduler is not None and hasattr(scheduler, 'step_iter'):
                scheduler.step()

        # Epoch 평균 손실
        avg_loss = epoch_loss / len(data_loader)

        # 손실 딕셔너리 평균
        for k in loss_dict_reduced:
            loss_dict_reduced[k] /= len(data_loader)

        # 학습률 스케줄러 (epoch마다 업데이트하는 경우)
        if scheduler is not None and not hasattr(scheduler, 'step_iter'):
            scheduler.step()

        print(f'\nEpoch {epoch} 완료:')
        print(f'  평균 손실: {avg_loss:.4f}')
        for k, v in loss_dict_reduced.items():
            print(f'  {k}: {v:.4f}')

        return avg_loss, loss_dict_reduced

    def train(self, train_loader, num_epochs, val_loader=None, scheduler=None, save_freq=1):
        """
        전체 학습 프로세스

        Args:
            train_loader: 학습 데이터 로더
            num_epochs: 총 Epoch 수
            val_loader: 검증 데이터 로더 (옵션)
            scheduler: 학습률 스케줄러 (옵션)
            save_freq: 체크포인트 저장 빈도

        Returns:
            history: 학습 히스토리
        """
        history = {
            'train_loss': [],
            'val_loss': []
        }

        best_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            # 학습
            train_loss, train_loss_dict = self.train_one_epoch(
                train_loader, epoch, scheduler
            )
            history['train_loss'].append(train_loss)

            # 검증
            if val_loader is not None:
                val_loss = self.validate(val_loader, epoch)
                history['val_loss'].append(val_loss)

                # Best 모델 저장
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_checkpoint(epoch, 'best_model.pth')
                    print(f'  Best 모델 저장 (val_loss: {val_loss:.4f})')

            # 주기적 체크포인트 저장
            if epoch % save_freq == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')

        # 최종 모델 저장
        self.save_checkpoint(num_epochs, 'final_model.pth')

        return history

    @torch.no_grad()
    def validate(self, data_loader, epoch):
        """
        검증

        Args:
            data_loader: 검증 데이터 로더
            epoch: 현재 Epoch

        Returns:
            avg_loss: 평균 검증 손실
        """
        self.model.train()  # 손실 계산을 위해 train 모드 유지

        val_loss = 0.0

        pbar = tqdm(data_loader, desc=f'Validation Epoch {epoch}')

        for images, targets in pbar:
            # 데이터를 디바이스로 이동
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Forward pass
            losses = self.model(images, targets)
            loss = losses['loss_total']

            val_loss += loss.item()

        avg_loss = val_loss / len(data_loader)
        print(f'\n검증 손실: {avg_loss:.4f}')

        return avg_loss

    def save_checkpoint(self, epoch, filename):
        """
        체크포인트 저장

        Args:
            epoch: 현재 Epoch
            filename: 저장 파일명
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """
        체크포인트 로드

        Args:
            filepath: 체크포인트 파일 경로

        Returns:
            epoch: 저장된 Epoch
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        print(f'체크포인트 로드 완료: Epoch {epoch}')

        return epoch
