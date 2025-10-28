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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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
        # 确保splits目录存在
        os.makedirs("data/splits", exist_ok=True)
    
    def _save_checkpoint(self, epoch, metric_value):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'backbone_state_dict': self.backbone.state_dict(),
            'objective_state_dict': self.objective.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metric_value': metric_value,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(
            self.config['output']['checkpoint_dir'], 
            f'checkpoint_epoch_{epoch:03d}.pth'
        )
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_split_info(self, idx_train, idx_test):
        """保存数据分割信息"""
        split_info = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'data_root': self.config['data']['root_dir'],
                'split_ratio': self.config['data']['split_ratio'],
                'seed': self.config['project']['seed']
            },
            'train_files': self.custom_dataset.df.loc[idx_train]['path'].tolist(),
            'val_files': self.custom_dataset.df.loc[idx_test]['path'].tolist(),
            'train_indices': idx_train.tolist(),
            'val_indices': idx_test.tolist(),
            'train_individuals': self.custom_dataset.df.loc[idx_train]['identity'].unique().tolist(),
            'val_individuals': self.custom_dataset.df.loc[idx_test]['identity'].unique().tolist()
        }
        
        # 使用时间戳和配置信息创建唯一的文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        split_filename = f"split_{timestamp}.yaml"
        split_path = os.path.join("data/splits", split_filename)
        
        with open(split_path, 'w') as f:
            yaml.dump(split_info, f, default_flow_style=False)
        
        print(f"数据分割信息已保存: {split_path}")
        return split_path
    
    def _validate_data_split(self, idx_train, idx_test):
        """验证数据分割的合理性"""
        print("验证数据分割...")
        
        # 检查训练集和验证集是否有重叠
        train_set = set(idx_train)
        val_set = set(idx_test)
        overlap = train_set.intersection(val_set)
        
        if overlap:
            raise ValueError(f"数据分割错误: 训练集和验证集有 {len(overlap)} 个重叠样本")
        
        # 检查每个个体在训练集和验证集中是否都有样本
        train_individuals = set(self.custom_dataset.df.loc[idx_train]['identity'].unique())
        val_individuals = set(self.custom_dataset.df.loc[idx_test]['identity'].unique())
        
        missing_in_val = train_individuals - val_individuals
        missing_in_train = val_individuals - train_individuals
        
        if missing_in_val:
            print(f"警告: 以下个体在验证集中缺失: {missing_in_val}")
        
        if missing_in_train:
            print(f"警告: 以下个体在训练集中缺失: {missing_in_train}")
        
        # 统计信息
        train_count = len(idx_train)
        val_count = len(idx_test)
        total_count = train_count + val_count
        
        print(f"数据分割验证通过")
        print(f"  训练集: {train_count} 图像, {len(train_individuals)} 个体")
        print(f"  验证集: {val_count} 图像, {len(val_individuals)} 个体")
        print(f"  分割比例: {train_count/total_count:.2%} 训练, {val_count/total_count:.2%} 验证")
        
        return True
    
    def prepare_data(self):
        """准备训练数据"""
        print("Preparing data...")
        
        # 创建自定义数据集
        self.custom_dataset = MyAnimalDataset(self.config['data']['root_dir'])
        
        # 分析数据集
        self.custom_dataset.analyze_dataset()
        
        # 数据分割
        splitter = splits.ClosedSetSplit(ratio_train=self.config['data']['split_ratio'])
        splits_list = splitter.split(self.custom_dataset.df)
        
        # 取第一个分割结果
        idx_train, idx_test = splits_list[0]
        
        # 验证数据分割
        self._validate_data_split(idx_train, idx_test)
        
        # 保存分割信息
        self.split_info_path = self._save_split_info(idx_train, idx_test)
        
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
        
        # 确保学习率和权重衰减是正确类型
        lr = float(self.config['training']['learning_rate'])
        weight_decay = float(self.config['training']['weight_decay'])
        
        if self.config['training']['optimizer'].lower() == 'adamw':
            self.optimizer = AdamW(
                params=params,
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = SGD(
                params=params,
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
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
        print(f"Learning rate: {lr}")
    
    def compute_validation_metrics(self):
        """计算验证集上的评估指标"""
        self.backbone.eval()
        
        # 提取特征
        feature_extractor = DeepFeatures(
            model=self.backbone,
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            device=str(self.device)
        )
        
        print("Extracting features for validation metrics...")
        train_features = feature_extractor(self.train_dataset)
        val_features = feature_extractor(self.val_dataset)
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(val_features.features, train_features.features)
        
        # 计算mAP
        average_precisions = []
        for i in range(len(self.val_dataset.labels)):
            query_label = self.val_dataset.labels[i]
            # 获取排序的索引（相似度从高到低）
            sorted_indices = np.argsort(similarity_matrix[i])[::-1]
            # 计算AP
            relevant = 0
            precision_at_k = []
            for k, idx in enumerate(sorted_indices):
                if self.train_dataset.labels[idx] == query_label:
                    relevant += 1
                    precision_at_k.append(relevant / (k + 1))
            if relevant > 0:
                ap = np.mean(precision_at_k)
                average_precisions.append(ap)
            else:
                average_precisions.append(0)
        
        mean_ap = np.mean(average_precisions)
        
        # 计算CMC曲线 (Rank-k 识别率)
        ranks = [1, 5, 10]
        cmc_scores = {}
        
        for rank in ranks:
            correct = 0
            for i in range(len(self.val_dataset.labels)):
                query_label = self.val_dataset.labels[i]
                sorted_indices = np.argsort(similarity_matrix[i])[::-1]
                # 检查前rank个结果中是否有正确匹配
                if any(self.train_dataset.labels[idx] == query_label for idx in sorted_indices[:rank]):
                    correct += 1
            cmc_scores[rank] = correct / len(self.val_dataset.labels)
        
        # 计算最近邻分类准确率
        knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
        knn.fit(train_features.features, self.train_dataset.labels)
        knn_predictions = knn.predict(val_features.features)
        knn_accuracy = accuracy_score(self.val_dataset.labels, knn_predictions)
        
        self.backbone.train()
        
        return {
            'mAP': mean_ap,
            'rank1': cmc_scores[1],
            'rank5': cmc_scores[5],
            'rank10': cmc_scores[10],
            'knn_accuracy': knn_accuracy
        }
    
    def create_trainer(self):
        """创建训练器"""
        
        # 初始化最佳指标跟踪
        self.best_mAP = 0.0
        self.best_rank1 = 0.0
        
        def epoch_callback(trainer, epoch_data):
            """训练回调函数 - 修正以正确获取训练损失"""
            # 正确获取训练损失
            train_loss = epoch_data.get('train_loss_epoch_avg', 0.0)
            
            # 从优化器获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 每5个epoch或在最后一个epoch计算验证指标
            validation_metrics = None
            if trainer.epoch % 5 == 0 or trainer.epoch == self.config['training']['epochs']:
                validation_metrics = self.compute_validation_metrics()
                
                # 更新最佳指标
                if validation_metrics['mAP'] > self.best_mAP:
                    self.best_mAP = validation_metrics['mAP']
                if validation_metrics['rank1'] > self.best_rank1:
                    self.best_rank1 = validation_metrics['rank1']
            
            # 格式化输出
            if validation_metrics:
                print(f"Epoch {trainer.epoch:3d}/{self.config['training']['epochs']} | "
                      f"LR: {current_lr:.2e} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val-mAP: {validation_metrics['mAP']:.4f} | "
                      f"Rank-1: {validation_metrics['rank1']:.4f} | "
                      f"Rank-5: {validation_metrics['rank5']:.4f} | "
                      f"KNN-Acc: {validation_metrics['knn_accuracy']:.4f}")
            else:
                print(f"Epoch {trainer.epoch:3d}/{self.config['training']['epochs']} | "
                      f"LR: {current_lr:.2e} | "
                      f"Train Loss: {train_loss:.4f}")
            
            # 保存检查点
            save_checkpoint = False
            if trainer.epoch % self.config['output']['save_frequency'] == 0:
                save_checkpoint = True
            
            # 如果是最后一个epoch，保存最终模型
            if trainer.epoch == self.config['training']['epochs']:
                save_checkpoint = True
            
            # 如果达到新的最佳mAP，也保存检查点
            if validation_metrics and validation_metrics['mAP'] == self.best_mAP:
                save_checkpoint = True
                print(f"  *** New best mAP: {validation_metrics['mAP']:.4f} ***")
            
            if save_checkpoint:
                metric_value = validation_metrics['mAP'] if validation_metrics else train_loss
                self._save_checkpoint(trainer.epoch, metric_value)
        
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
        print("Validation metrics will be computed every 5 epochs")
        
        # 开始训练
        trainer.train()
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # 输出最终评估结果
        print("\n" + "=" * 60)
        print("Final Evaluation Results")
        print("=" * 60)
        final_metrics = self.compute_validation_metrics()
        print(f"Mean Average Precision (mAP): {final_metrics['mAP']:.4f}")
        print(f"Rank-1 Accuracy: {final_metrics['rank1']:.4f}")
        print(f"Rank-5 Accuracy: {final_metrics['rank5']:.4f}")
        print(f"Rank-10 Accuracy: {final_metrics['rank10']:.4f}")
        print(f"KNN Classification Accuracy: {final_metrics['knn_accuracy']:.4f}")
        print(f"Best mAP during training: {self.best_mAP:.4f}")
        print(f"Best Rank-1 during training: {self.best_rank1:.4f}")
        
        print("\nTraining completed!")
        print(f"Total duration: {training_duration}")
        
        # 保存最终模型
        final_checkpoint = {
            'backbone_state_dict': self.backbone.state_dict(),
            'objective_state_dict': self.objective.state_dict(),
            'num_classes': num_classes,
            'config': self.config,
            'training_duration': str(training_duration),
            'completed_at': datetime.now().isoformat(),
            'final_metrics': final_metrics,
            'best_mAP': self.best_mAP,
            'best_rank1': self.best_rank1
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