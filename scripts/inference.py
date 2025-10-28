#!/usr/bin/env python3
"""
MegaDescriptor 模型推理脚本 - 增强版本
支持批量或单个图像推理，可视化结果并保存，包括Top3匹配图像
"""

import sys
import os
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import json

from wildlife_tools.data.dataset import ImageDataset
from wildlife_tools.features.deep import DeepFeatures
from src.data.transforms import get_val_transforms
from src.data.datasets import MyAnimalDataset

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy数据类型"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

class MegaDescriptorInference:
    """MegaDescriptor 推理器"""
    
    def __init__(self, checkpoint_path, config_path, device='auto'):
        """
        初始化推理器
        
        Args:
            checkpoint_path: 模型检查点路径
            config_path: 配置文件路径
            device: 推理设备 ('auto', 'cuda', 'cpu')
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = self._setup_device(device)
        
        # 加载配置和模型
        self.config = self._load_config()
        self.model = self._load_model()
        self.transform = get_val_transforms(self.config)
        
        # 准备gallery数据集（训练集）
        self._prepare_gallery()
        
        print(f"推理器初始化完成")
        print(f"设备: {self.device}")
        print(f"Gallery大小: {len(self.gallery_dataset)} 图像")
        print(f"类别数: {self.gallery_dataset.num_classes}")
    
    def _setup_device(self, device):
        """设置推理设备"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self):
        """加载模型"""
        print(f"加载模型: {self.checkpoint_path}")
        
        # 修复：添加 weights_only=False 参数
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        # 创建骨干网络
        model = timm.create_model(
            'hf-hub:BVRA/MegaDescriptor-T-224',
            num_classes=0,
            pretrained=False
        )
        
        # 加载权重
        model.load_state_dict(checkpoint['backbone_state_dict'])
        model.eval()
        model = model.to(self.device)
        
        print("模型加载成功")
        if 'final_metrics' in checkpoint:
            print(f"模型训练指标 - mAP: {checkpoint['final_metrics']['mAP']:.4f}")
        
        return model
    
    def _prepare_gallery(self):
        """准备gallery数据集（训练集）"""
        print("准备gallery数据集...")
        
        # 创建完整数据集
        custom_dataset = MyAnimalDataset(self.config['data']['root_dir'])
        
        # 重现训练时的数据分割
        from wildlife_datasets import splits
        splitter = splits.ClosedSetSplit(ratio_train=self.config['data']['split_ratio'])
        splits_list = splitter.split(custom_dataset.df)
        idx_train, _ = splits_list[0]
        
        df_train = custom_dataset.df.loc[idx_train]
        
        # 创建gallery数据集
        self.gallery_dataset = ImageDataset(
            metadata=df_train,
            root=self.config['data']['root_dir'],
            transform=self.transform,
            col_path='path',
            col_label='identity'
        )
        
        # 提取gallery特征
        feature_extractor = DeepFeatures(
            model=self.model,
            batch_size=self.config['data']['batch_size'],
            device=str(self.device)
        )
        
        print("提取gallery特征...")
        self.gallery_features = feature_extractor(self.gallery_dataset)
        
        # 训练KNN分类器
        self.knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
        self.knn.fit(self.gallery_features.features, self.gallery_dataset.labels)
        
        print(f"Gallery特征提取完成: {self.gallery_features.features.shape}")
    
    def _load_and_transform_image(self, image_path):
        """加载并预处理图像"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor.unsqueeze(0)  # 添加batch维度
        except Exception as e:
            print(f"图像加载失败 {image_path}: {e}")
            return None
    
    def _safe_cosine_similarity(self, query_features, gallery_features):
        """安全的余弦相似度计算，避免数值问题"""
        try:
            # 标准化特征向量以避免数值问题
            query_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-10)
            gallery_norm = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-10)
            
            # 计算余弦相似度
            similarities = np.dot(query_norm, gallery_norm.T)
            
            # 确保相似度在合理范围内
            similarities = np.clip(similarities, -1.0, 1.0)
            
            return similarities[0]  # 返回第一个查询的结果
        except Exception as e:
            print(f"余弦相似度计算错误: {e}")
            # 返回默认相似度
            return np.ones(len(gallery_features)) * 0.5
    
    def extract_features(self, image_tensor):
        """提取图像特征"""
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            features = self.model(image_tensor)
            return features.cpu().numpy()
    
    def predict_single(self, image_path, top_k=5):
        """
        对单个图像进行预测
        
        Args:
            image_path: 图像路径
            top_k: 返回前k个最相似结果
            
        Returns:
            dict: 预测结果
        """
        # 加载和预处理图像
        image_tensor = self._load_and_transform_image(image_path)
        if image_tensor is None:
            return None
        
        # 提取特征
        query_features = self.extract_features(image_tensor)
        
        # 使用安全的余弦相似度计算
        similarities = self._safe_cosine_similarity(query_features, self.gallery_features.features)
        
        # 获取top-k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        top_labels = [self.gallery_dataset.labels[i] for i in top_indices]
        top_paths = [self.gallery_dataset.metadata.iloc[i]['path'] for i in top_indices]
        
        # KNN预测
        try:
            knn_prediction = self.knn.predict(query_features)[0]
            knn_confidence = np.max(self.knn.predict_proba(query_features))
        except Exception as e:
            print(f"KNN预测错误: {e}")
            # 使用相似度最高的结果作为预测
            knn_prediction = top_labels[0]
            knn_confidence = top_similarities[0]
        
        # 获取预测个体在gallery中的所有图像
        individual_images = []
        individual_indices = np.where(self.gallery_dataset.labels == knn_prediction)[0]
        for idx in individual_indices[:10]:  # 最多显示10张
            individual_images.append({
                'path': self.gallery_dataset.metadata.iloc[idx]['path'],
                'similarity': float(similarities[idx])  # 转换为Python float
            })
        
        # 确保所有数值都是Python原生类型
        result = {
            'query_image': image_path,
            'predicted_individual': str(knn_prediction),  # 确保是字符串
            'confidence': float(knn_confidence),
            'top_matches': [
                {
                    'rank': int(i + 1),
                    'individual': str(top_labels[i]),
                    'similarity': float(top_similarities[i]),
                    'image_path': str(top_paths[i]),
                    'full_path': os.path.join(self.config['data']['root_dir'], str(top_paths[i]))
                }
                for i in range(len(top_indices))
            ],
            'individual_gallery': individual_images
        }
        
        return result
    
    def predict_batch(self, image_paths, top_k=5):
        """
        批量预测
        
        Args:
            image_paths: 图像路径列表
            top_k: 返回前k个最相似结果
            
        Returns:
            list: 每个图像的预测结果
        """
        results = []
        for i, image_path in enumerate(image_paths):
            print(f"处理图像 {i+1}/{len(image_paths)}: {image_path}")
            result = self.predict_single(image_path, top_k)
            if result is not None:
                results.append(result)
        
        return results
    
    def save_top_images(self, results, output_dir):
        """
        保存Top3匹配图像到指定目录
        
        Args:
            results: 推理结果列表
            output_dir: 输出目录
        """
        img_dir = os.path.join(output_dir, 'img')
        os.makedirs(img_dir, exist_ok=True)
        
        for i, result in enumerate(results):
            # 创建查询图像的子目录
            query_name = Path(result['query_image']).stem
            query_dir = os.path.join(img_dir, query_name)
            os.makedirs(query_dir, exist_ok=True)
            
            # 保存查询图像
            query_src = result['query_image']
            query_dst = os.path.join(query_dir, f'query_{Path(query_src).name}')
            shutil.copy2(query_src, query_dst)
            print(f"查询图像已保存: {query_dst}")
            
            # 保存Top3匹配图像
            for match in result['top_matches'][:3]:  # 只保存前3个
                rank = match['rank']
                src_path = match['full_path']
                
                # 确保源文件存在
                if not os.path.exists(src_path):
                    print(f"警告: 匹配图像不存在: {src_path}")
                    continue
                
                # 生成目标文件名
                dst_filename = f'top{rank}_{match["individual"]}_{Path(src_path).name}'
                dst_path = os.path.join(query_dir, dst_filename)
                
                # 复制图像
                shutil.copy2(src_path, dst_path)
                print(f"Top{rank}匹配图像已保存: {dst_path}")
    
    def visualize_result(self, result, save_path=None, show=True):
        """
        可视化推理结果
        
        Args:
            result: 预测结果
            save_path: 保存路径（可选）
            show: 是否显示图像
        """
        if result is None:
            print("无有效结果可可视化")
            return
        
        # 创建图像网格
        fig, axes = plt.subplots(2, 6, figsize=(20, 8))
        fig.suptitle(f'推理结果 - 预测个体: {result["predicted_individual"]} (置信度: {result["confidence"]:.3f})', 
                    fontsize=16, fontweight='bold')
        
        # 清空所有子图
        for ax in axes.flat:
            ax.axis('off')
        
        # 显示查询图像
        query_img = Image.open(result['query_image']).convert('RGB')
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title(f'查询图像\n{Path(result["query_image"]).name}', fontsize=10)
        axes[0, 0].axis('on')
        
        # 显示前5个匹配结果
        for i, match in enumerate(result['top_matches']):
            if i >= 5:  # 只显示前5个
                break
            row = i // 3
            col = (i % 3) + 1
            
            match_path = os.path.join(self.config['data']['root_dir'], match['image_path'])
            match_img = Image.open(match_path).convert('RGB')
            axes[row, col].imshow(match_img)
            axes[row, col].set_title(f'#{match["rank"]}: {match["individual"]}\n相似度: {match["similarity"]:.3f}', 
                                   fontsize=9, color='green' if match['rank'] == 1 else 'black')
            axes[row, col].axis('on')
        
        # 显示预测个体的其他图像
        individual_imgs = result['individual_gallery'][:5]  # 最多显示5张
        for i, img_info in enumerate(individual_imgs):
            img_path = os.path.join(self.config['data']['root_dir'], img_info['path'])
            img = Image.open(img_path).convert('RGB')
            axes[1, i+1].imshow(img)
            axes[1, i+1].set_title(f'同类图像 {i+1}\n相似度: {img_info["similarity"]:.3f}', fontsize=9)
            axes[1, i+1].axis('on')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_results(self, results, output_dir):
        """
        保存推理结果到CSV文件
        
        Args:
            results: 推理结果列表
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备CSV数据
        csv_data = []
        for result in results:
            csv_data.append({
                'query_image': result['query_image'],
                'predicted_individual': result['predicted_individual'],
                'confidence': result['confidence'],
                'top1_individual': result['top_matches'][0]['individual'],
                'top1_similarity': result['top_matches'][0]['similarity'],
                'top2_individual': result['top_matches'][1]['individual'] if len(result['top_matches']) > 1 else '',
                'top2_similarity': result['top_matches'][1]['similarity'] if len(result['top_matches']) > 1 else 0,
                'top3_individual': result['top_matches'][2]['individual'] if len(result['top_matches']) > 2 else '',
                'top3_similarity': result['top_matches'][2]['similarity'] if len(result['top_matches']) > 2 else 0,
            })
        
        # 保存CSV
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, 'inference_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"推理结果已保存到: {csv_path}")
        
        # 保存详细结果 - 使用自定义JSON编码器
        detailed_path = os.path.join(output_dir, 'detailed_results.json')
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"详细结果已保存到: {detailed_path}")
        
        # 保存Top3匹配图像
        self.save_top_images(results, output_dir)
        
        return csv_path, detailed_path


def main():
    parser = argparse.ArgumentParser(description='MegaDescriptor 模型推理')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像路径或目录路径')
    parser.add_argument('--output_dir', type=str, default='outputs/inference',
                       help='输出目录路径')
    parser.add_argument('--top_k', type=int, default=5,
                       help='返回前k个最相似结果')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批量推理时的批次大小')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='推理设备')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化结果')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='是否保存可视化图像')
    
    args = parser.parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.input):
        print(f"错误: 输入路径不存在: {args.input}")
        return
    
    # 初始化推理器
    print("初始化推理器...")
    inference = MegaDescriptorInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # 准备输入图像列表
    if os.path.isfile(args.input):
        image_paths = [args.input]
        print(f"单图像推理: {args.input}")
    else:
        # 支持常见图像格式
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.nef']
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(args.input).glob(f'**/{ext}'))
            image_paths.extend(Path(args.input).glob(f'**/{ext.upper()}'))
        
        image_paths = [str(p) for p in image_paths]
        print(f"批量推理: 找到 {len(image_paths)} 张图像")
    
    if not image_paths:
        print("错误: 未找到任何图像文件")
        return
    
    # 执行推理
    print("开始推理...")
    results = inference.predict_batch(image_paths, top_k=args.top_k)
    
    if not results:
        print("错误: 推理未产生任何结果")
        return
    
    print(f"推理完成: 成功处理 {len(results)} 张图像")
    
    # 保存结果
    print("保存推理结果...")
    csv_path, detailed_path = inference.save_results(results, args.output_dir)
    
    # 可视化和保存可视化结果
    if args.visualize or args.save_visualizations:
        print("生成可视化结果...")
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        for i, result in enumerate(results):
            save_path = None
            if args.save_visualizations:
                query_name = Path(result['query_image']).stem
                save_path = os.path.join(vis_dir, f'{query_name}_result.png')
            
            inference.visualize_result(
                result, 
                save_path=save_path, 
                show=args.visualize and i == 0  # 只显示第一张图像
            )
    
    print("推理流程完成!")
    print(f"结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()