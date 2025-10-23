#!/usr/bin/env python3
"""
模型评估脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import timm
from wildlife_tools.features import DeepFeatures
from wildlife_datasets.data import ImageDataset
from src.data.transforms import get_val_transforms
import yaml

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载模型
    print(f"Loading model: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # 创建骨干网络
    backbone = timm.create_model(
        config['model']['name'],
        num_classes=0,
        pretrained=False
    )
    
    # 加载权重
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    backbone.eval()
    
    print("Model loaded successfully")
    print(f"Training epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Validation accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
    
    # 准备数据
    from src.data.datasets import MyAnimalDataset
    custom_dataset = MyAnimalDataset(config['data']['root_dir'])
    
    # 使用全部数据进行评估
    transform = get_val_transforms(config)
    eval_dataset = ImageDataset(
        metadata=custom_dataset.df,
        root=config['data']['root_dir'],
        transform=transform,
        col_path='path',
        col_label='identity'
    )
    
    # 提取特征
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_extractor = DeepFeatures(
        model=backbone,
        batch_size=config['data']['batch_size'],
        device=device
    )
    
    print("Extracting features for evaluation...")
    features = feature_extractor(eval_dataset)
    
    print("Evaluation completed")
    print(f"Feature shape: {features.features.shape}")
    print(f"Number of individuals: {eval_dataset.num_classes}")

if __name__ == '__main__':
    main()