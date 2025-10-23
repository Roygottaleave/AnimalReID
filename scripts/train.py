#!/usr/bin/env python3
"""
MegaDescriptor微调主脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.training.trainer import MegaDescriptorFinetuner

def main():
    parser = argparse.ArgumentParser(description='Fine-tune MegaDescriptor model')
    parser.add_argument('--config', type=str, default='configs/train_megadescriptor.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"Configuration file does not exist: {args.config}")
        return
    
    # 初始化训练器
    print(f"Using configuration file: {args.config}")
    trainer = MegaDescriptorFinetuner(args.config)
    
    # 开始训练
    try:
        final_model_path = trainer.train()
        print(f"Training completed successfully! Model saved at: {final_model_path}")
    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == '__main__':
    main()