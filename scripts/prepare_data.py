#!/usr/bin/env python3
"""
数据准备和验证脚本 - 包含数据分割验证功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import pandas as pd
from wildlife_datasets import splits
from src.data.datasets import MyAnimalDataset

def validate_data_accessibility(data_root, dataset):
    """验证数据文件可访问性"""
    print("检查文件可访问性...")
    missing_files = []
    accessible_files = 0
    
    for idx, row in dataset.df.iterrows():
        full_path = os.path.join(data_root, row['path'])
        if os.path.exists(full_path):
            accessible_files += 1
        else:
            missing_files.append(full_path)
    
    print(f"可访问文件: {accessible_files}/{len(dataset.df)}")
    if missing_files:
        print(f"缺失文件: {len(missing_files)}")
        for i, missing_file in enumerate(missing_files[:5]):
            print(f"  {i+1}. {missing_file}")
        if len(missing_files) > 5:
            print(f"  ... 还有 {len(missing_files) - 5} 个缺失文件")
        return False
    else:
        print("所有文件都可访问")
        return True

def validate_data_split(data_root, split_ratio=0.8, seed=42):
    """验证数据分割的合理性"""
    print("\n验证数据分割...")
    
    # 创建数据集
    dataset = MyAnimalDataset(data_root)
    
    # 数据分割
    splitter = splits.ClosedSetSplit(ratio_train=split_ratio)
    splits_list = splitter.split(dataset.df)
    idx_train, idx_test = splits_list[0]
    
    # 检查训练集和验证集是否有重叠
    train_set = set(idx_train)
    val_set = set(idx_test)
    overlap = train_set.intersection(val_set)
    
    if overlap:
        print(f"错误: 训练集和验证集有 {len(overlap)} 个重叠样本")
        return False
    
    # 检查每个个体在训练集和验证集中是否都有样本
    train_individuals = set(dataset.df.loc[idx_train]['identity'].unique())
    val_individuals = set(dataset.df.loc[idx_test]['identity'].unique())
    
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

def validate_existing_split(data_root, split_file):
    """验证已有的分割文件"""
    print(f"验证分割文件: {split_file}")
    
    if not os.path.exists(split_file):
        print(f"分割文件不存在: {split_file}")
        return False
    
    # 加载分割信息
    with open(split_file, 'r') as f:
        split_info = yaml.safe_load(f)
    
    # 创建数据集
    dataset = MyAnimalDataset(data_root)
    
    # 检查文件是否存在
    all_files_valid = True
    for file_list_name in ['train_files', 'val_files']:
        if file_list_name in split_info:
            for file_path in split_info[file_list_name]:
                full_path = os.path.join(data_root, file_path)
                if not os.path.exists(full_path):
                    print(f"文件不存在: {full_path}")
                    all_files_valid = False
    
    if not all_files_valid:
        print("分割文件中包含不存在的文件")
        return False
    
    print("分割文件验证通过")
    print(f"  训练文件: {len(split_info.get('train_files', []))}")
    print(f"  验证文件: {len(split_info.get('val_files', []))}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='准备和验证训练数据')
    parser.add_argument('--data_root', type=str, help='数据根目录路径')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--validate_split', action='store_true', 
                       help='验证数据分割的合理性')
    parser.add_argument('--split_file', type=str, 
                       help='验证已有的分割文件')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                       help='训练集分割比例 (默认: 0.8)')
    
    args = parser.parse_args()
    
    # 参数验证
    if not args.data_root and not args.config:
        parser.error("必须提供 --data_root 或 --config 参数")
    
    try:
        # 确定数据根目录
        if args.config:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            data_root = config['data']['root_dir']
        else:
            data_root = args.data_root
        
        print(f"检查数据目录: {data_root}")
        
        if not os.path.exists(data_root):
            print(f"数据目录不存在: {data_root}")
            return
        
        # 创建数据集实例
        dataset = MyAnimalDataset(data_root)
        print("数据集加载成功!")
        
        # 分析数据集
        dataset.analyze_dataset()
        
        # 验证文件可访问性
        is_accessible = validate_data_accessibility(data_root, dataset)
        
        if not is_accessible:
            print("数据验证失败: 存在无法访问的文件")
            return
        
        # 验证数据分割（如果请求）
        if args.validate_split:
            split_valid = validate_data_split(data_root, args.split_ratio)
            if not split_valid:
                print("数据分割验证失败")
                return
        
        # 验证已有的分割文件（如果提供）
        if args.split_file:
            split_file_valid = validate_existing_split(data_root, args.split_file)
            if not split_file_valid:
                print("分割文件验证失败")
                return
        
        print("\n所有验证通过! 数据准备完成")
            
    except Exception as e:
        print(f"数据准备失败: {e}")
        raise

if __name__ == '__main__':
    main()