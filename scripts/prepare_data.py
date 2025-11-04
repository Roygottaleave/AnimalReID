#!/usr/bin/env python3
"""
数据准备和验证脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.data.datasets import MyAnimalDataset

def main():
    parser = argparse.ArgumentParser(description='Prepare and validate data')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to data root directory')
    
    args = parser.parse_args()
    
    print(f"Checking data directory: {args.data_root}")
    
    if not os.path.exists(args.data_root):
        print(f"Data directory does not exist: {args.data_root}")
        return
    
    # 创建数据集实例
    try:
        dataset = MyAnimalDataset(args.data_root)
        print("Dataset loaded successfully!")
        
        # 分析数据集
        dataset.analyze_dataset()
        
        # 检查文件是否存在
        print("Checking file accessibility...")
        missing_files = []
        accessible_files = 0
        
        for idx, row in dataset.df.iterrows():
            full_path = os.path.join(args.data_root, row['path'])
            if os.path.exists(full_path):
                accessible_files += 1
            else:
                missing_files.append(full_path)
        
        print(f"Accessible files: {accessible_files}/{len(dataset.df)}")
        if missing_files:
            print(f"Missing files: {len(missing_files)}")
            for i, missing_file in enumerate(missing_files[:5]):
                print(f"  {i+1}. {missing_file}")
            if len(missing_files) > 5:
                print(f"  ... and {len(missing_files) - 5} more missing files")
        else:
            print("All files are accessible")
            
    except Exception as e:
        print(f"Data loading failed: {e}")
        raise

if __name__ == '__main__':
    main()