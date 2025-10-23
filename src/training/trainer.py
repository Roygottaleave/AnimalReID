import timm
import itertools
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from wildlife_tools.train import ArcFaceLoss, BasicTrainer, set_seed
from wildlife_datasets import splits
from wildlife_tools.data.dataset import ImageDataset
from wildlife_tools.features.deep import DeepFeatures
import os
from datetime import datetime
import yaml
import pandas as pd

from src.data.datasets import MyAnimalDataset
from src.data.transforms import get_train_transforms, get_val_transforms

class MegaDescriptorFinetuner:
    """MegaDescriptor微调器"""
    
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self._setup_directories()
        set_seed(self.config['project']['seed'])
        
    def _load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_directories(self):
        """创建输出目录"""
        os.makedirs(self.config['output']['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['output']['log_dir'], exist_ok=True)
    
    def prepare_data(self):
        """准备训练数据"""

        
        print("Preparing data...")
        
        # 创建自定义数据集
        self.custom_dataset = MyAnimalDataset(self.config['data']['root_dir'])
        
        # 分析数据集
        self.custom_dataset.analyze_dataset()
        
        # 数据分割 - 修复这里
        splitter = splits.ClosedSetSplit(ratio_train=self.config['data']['split_ratio'])
        
        # split() 返回的是列表，不是迭代器
        splits_list = splitter.split(self.custom_dataset.df)
        
        # 取第一个分割结果
        idx_train, idx_test = splits_list[0]
        
        self.df_train = self.custom_dataset.df.loc[idx_train]
        self.df_test = self.custom_dataset.df.loc[idx_test]
        
        # 创建数据变换
        train_transform = get_train_transforms(self.config)
        val_transform = get_val_transforms(self.config)
        
        # 创建ImageDataset
        self.train_dataset = ImageDataset(
            metadata=self.df_train,
            root=self.config['data']['root_dir'],
            transform=train_transform,
            col_path='path',
            col_label='identity'
        )
        
        self.val_dataset = ImageDataset(
            metadata=self.df_test,
            root=self.config['data']['root_dir'],
            transform=val_transform,
            col_path='path',
            col_label='identity'
        )
        
        print("Data preparation completed")
        print(f"Training set: {len(self.train_dataset)} images, {self.train_dataset.num_classes} individuals")
        print(f"Validation set: {len(self.val_dataset)} images, {self.val_dataset.num_classes} individuals")
        
        return self.train_dataset.num_classes
    
    def setup_model(self, num_classes):
        """设置模型、损失函数和优化器"""
        print("Initializing model...")
        
        # 加载MegaDescriptor骨干网络
        '''
        self.backbone = timm.create_model(
            self.config['model']['name'],
            num_classes=0,
            pretrained=self.config['model']['pretrained']
        )
        '''
        self.backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True)
        
        # 创建ArcFace损失函数
        loss_config = self.config['training']['loss']
        self.objective = ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=loss_config['embedding_size'],
            margin=loss_config['margin'],
            scale=loss_config['scale']
        )
        
        # 设置优化器
        params = itertools.chain(self.backbone.parameters(), self.objective.parameters())
        
        if self.config['training']['optimizer'].lower() == 'adamw':
            self.optimizer = AdamW(
                params=params,
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            self.optimizer = SGD(
                params=params,
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['training']['weight_decay']
            )
        
        # 学习率调度器
        if self.config['training']['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['training']['epochs']
            )
        else:
            self.scheduler = None
        
        print("Model initialization completed")
        print(f"Backbone: {self.config['model']['name']}")
        print(f"Number of classes: {num_classes}")
        print(f"Optimizer: {self.config['training']['optimizer']}")
        print(f"Learning rate: {self.config['training']['learning_rate']}")
    
    def create_trainer(self):
        """创建训练器"""
        
        def epoch_callback(trainer, epoch_data):
            """训练回调函数 - 修正以正确获取训练损失"""
            # 打印完整的epoch_data用于调试
            print(f"DEBUG: epoch_data = {epoch_data}")
            
            # 正确获取训练损失
            train_loss = epoch_data.get('train_loss_epoch_avg', 0.0)
            
            # 从优化器获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 由于BasicTrainer可能不提供验证信息，我们只显示训练损失
            # 如果需要验证，可能需要手动添加验证步骤
            print(f"Epoch {trainer.epoch:3d}/{self.config['training']['epochs']} | "
                  f"LR: {current_lr:.2e} | "
                  f"Train Loss: {train_loss:.4f}")
            
            # 保存检查点
            if trainer.epoch % self.config['output']['save_frequency'] == 0 or trainer.epoch == self.config['training']['epochs']:
                self._save_checkpoint(trainer.epoch, train_loss)  # 使用训练损失作为保存依据
        
        # 创建BasicTrainer
        self.trainer = BasicTrainer(
            dataset=self.train_dataset,
            model=self.backbone,
            objective=self.objective,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epochs=self.config['training']['epochs'],
            device=self.device,
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            accumulation_steps=self.config['training']['accumulation_steps'],
            epoch_callback=epoch_callback
        )
        
        return self.trainer
        
        def _save_checkpoint(self, epoch, val_accuracy):
            """保存检查点"""
            checkpoint = {
                'epoch': epoch,
                'backbone_state_dict': self.backbone.state_dict(),
                'objective_state_dict': self.objective.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'val_accuracy': val_accuracy,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            checkpoint_path = os.path.join(
                self.config['output']['checkpoint_dir'], 
                f'checkpoint_epoch_{epoch:03d}.pth'
            )
            
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """执行训练"""
        print("=" * 60)
        print("MegaDescriptor Fine-tuning Training")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # 准备数据
        num_classes = self.prepare_data()
        
        # 设置模型
        self.setup_model(num_classes)
        
        # 创建训练器
        trainer = self.create_trainer()
        
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config['training']['epochs']}")
        print(f"Batch size: {self.config['data']['batch_size']}")
        print(f"Gradient accumulation: {self.config['training']['accumulation_steps']}")
        
        # 开始训练
        trainer.train()
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        print("Training completed!")
        print(f"Total duration: {training_duration}")
        
        # 保存最终模型
        final_checkpoint = {
            'backbone_state_dict': self.backbone.state_dict(),
            'objective_state_dict': self.objective.state_dict(),
            'num_classes': num_classes,
            'config': self.config,
            'training_duration': str(training_duration),
            'completed_at': datetime.now().isoformat()
        }
        
        final_path = os.path.join(self.config['output']['checkpoint_dir'], 'final_model.pth')
        torch.save(final_checkpoint, final_path)
        print(f"Final model saved: {final_path}")
        
        return final_path
    
    def extract_features(self, dataset=None):
        """使用训练好的模型提取特征"""
        if dataset is None:
            dataset = self.val_dataset
        
        print("Extracting features...")
        
        feature_extractor = DeepFeatures(
            model=self.backbone,
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            device=str(self.device)
        )
        
        features = feature_extractor(dataset)
        print(f"Feature extraction completed: {features.features.shape}")
        
        return features