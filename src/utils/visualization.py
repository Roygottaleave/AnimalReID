#!/usr/bin/env python3
# src/utils/visualization.py

import sys
import os

# 修复路径问题
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
project_root_parent = os.path.dirname(project_root)

if project_root_parent not in sys.path:
    sys.path.insert(0, project_root_parent)

import torch
import timm
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TSNEVisualizer:
    """
    t-SNE可视化工具类
    """
    
    def __init__(self, config_path, checkpoint_path, device='auto'):
        self.config = self._load_config(config_path)
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = self._load_model(checkpoint_path)
        
        # 延迟导入项目特定模块
        try:
            from src.data.transforms import get_val_transforms
            self.transform = get_val_transforms(self.config)
        except ImportError as e:
            logger.error(f"Failed to import project modules: {e}")
            raise
        
        logger.info(f"TSNEVisualizer initialized with device: {self.device}")
    
    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        logger.info(f"Loading model from: {checkpoint_path}")
        
        model = timm.create_model(
            self.config['model']['name'],
            num_classes=0,
            pretrained=False
        )
        
        # 处理 PyTorch 2.6 的 weights_only 问题
        checkpoint = self._safe_load_checkpoint(checkpoint_path)
        
        if 'backbone_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['backbone_state_dict'])
            logger.info("Loaded backbone weights from checkpoint")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded model weights from checkpoint")
        else:
            # 尝试直接加载
            try:
                model.load_state_dict(checkpoint)
                logger.info("Loaded model weights from checkpoint (direct)")
            except:
                # 如果直接加载失败，可能是整个模型被保存了
                if hasattr(checkpoint, 'keys') and 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                    logger.info("Loaded model weights from checkpoint['model']")
                else:
                    raise ValueError("Could not load model weights from checkpoint")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _safe_load_checkpoint(self, checkpoint_path):
        """安全加载checkpoint，处理PyTorch 2.6的weights_only问题"""
        try:
            # 首先尝试使用 weights_only=True (更安全)
            return torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception as e:
            logger.warning(f"Failed to load with weights_only=True: {e}")
            logger.warning("Trying with weights_only=False (make sure you trust the checkpoint source)")
            
            try:
                # 如果失败，使用 weights_only=False
                return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            except Exception as e2:
                logger.error(f"Also failed to load with weights_only=False: {e2}")
                raise RuntimeError(f"Could not load checkpoint: {e2}")
    
    def extract_features_custom(self, dataset_path, max_samples=None):
        """自定义特征提取方法，避免DeepFeatures的问题"""
        logger.info(f"Extracting features from: {dataset_path}")
        
        # 延迟导入避免冲突
        try:
            from src.data.datasets import MyAnimalDataset
            from wildlife_tools.data.dataset import ImageDataset
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise
        
        # 创建数据集 - 使用ImageDataset确保正确的数据格式
        my_dataset = MyAnimalDataset(dataset_path)
        dataset = ImageDataset(
            metadata=my_dataset.df,
            root=dataset_path,
            transform=self.transform,
            col_path='path',
            col_label='identity'
        )
        
        if max_samples and len(dataset) > max_samples:
            logger.info(f"Sampling {max_samples} from {len(dataset)} total samples")
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            from torch.utils.data import Subset
            dataset = Subset(dataset, indices)
            # 对于Subset，我们需要手动获取标签
            labels = [dataset.dataset.labels[i] for i in indices]
        else:
            labels = dataset.labels
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            shuffle=False
        )
        
        # 手动提取特征
        self.model.eval()
        all_features = []
        all_labels = []
        
        logger.info("Starting feature extraction...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                if isinstance(batch, (list, tuple)):
                    images, batch_labels = batch
                else:
                    images = batch
                    batch_labels = None
                
                images = images.to(self.device)
                features = self.model(images)
                
                if isinstance(features, tuple):
                    features = features[0]
                
                all_features.append(features.cpu().numpy())
                if batch_labels is not None:
                    all_labels.extend(batch_labels.numpy())
        
        # 合并所有特征
        embeddings = np.vstack(all_features)
        
        # 如果没有从batch获取标签，使用数据集的标签
        if len(all_labels) == 0:
            all_labels = labels
        
        # 获取标签名称
        label_names = my_dataset.df['identity'].values
        image_paths = my_dataset.df['path'].values if 'path' in my_dataset.df.columns else None
        
        logger.info(f"Extracted {len(embeddings)} features with {len(np.unique(all_labels))} classes")
        return embeddings, np.array(all_labels), label_names, image_paths
    
    def compute_tsne(self, embeddings, n_components=2, perplexity=30, 
                    random_state=42, use_pca=True, pca_components=50):
        logger.info("Computing t-SNE...")
        
        # 更安全的标准化，避免数值问题
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # 处理可能的数值问题
        embeddings_scaled = np.nan_to_num(embeddings_scaled, nan=0.0, posinf=1e3, neginf=-1e3)
        
        if use_pca and embeddings_scaled.shape[1] > pca_components:
            logger.info(f"Applying PCA pre-processing ({embeddings_scaled.shape[1]} -> {pca_components} dimensions)")
            pca = PCA(n_components=pca_components, random_state=random_state)
            embeddings_reduced = pca.fit_transform(embeddings_scaled)
            explained_variance = pca.explained_variance_ratio_.sum()
            logger.info(f"PCA explained variance: {explained_variance:.3f}")
        else:
            embeddings_reduced = embeddings_scaled
        
        # 修复 TSNE 参数：n_iter -> max_iter
        try:
            # 尝试使用新参数名 max_iter
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                max_iter=1000,  # 新版本参数名
                learning_rate='auto',
                init='random'
            )
        except TypeError:
            # 如果失败，回退到旧参数名 n_iter
            logger.warning("Using deprecated parameter 'n_iter' for TSNE")
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                n_iter=1000,  # 旧版本参数名
                learning_rate='auto',
                init='random'
            )
        
        embeddings_tsne = tsne.fit_transform(embeddings_reduced)
        logger.info("t-SNE computation completed")
        return embeddings_tsne
    
    def plot_tsne(self, embeddings_tsne, labels, label_names, 
                 title="t-SNE Visualization", save_path=None, figsize=(12, 10)):
        logger.info("Plotting t-SNE visualization...")
        
        plt.figure(figsize=figsize)
        
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            
            # 修复标签名称获取逻辑
            if len(label_names) > 0:
                # 找到第一个属于该类别的样本索引
                first_idx = np.where(mask)[0][0] if np.any(mask) else 0
                legend_label = str(label_names[first_idx])
            else:
                legend_label = f'Class {label}'
                
            plt.scatter(
                embeddings_tsne[mask, 0], 
                embeddings_tsne[mask, 1],
                c=[colors[i]], 
                label=legend_label,
                alpha=0.7,
                s=50,
                edgecolors='w',
                linewidth=0.5
            )
        
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        
        plt.legend(
            bbox_to_anchor=(1.05, 1), 
            loc='upper left',
            fontsize=10,
            frameon=True,
            fancybox=True,
            framealpha=0.8
        )
        
        plt.grid(True, alpha=0.3)
        
        stats_text = f'Samples: {len(embeddings_tsne)}\nClasses: {len(unique_labels)}'
        plt.figtext(
            0.02, 0.02, 
            stats_text, 
            fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        return True
    
    def visualize_dataset(self, dataset_path, output_dir="outputs/visualizations", 
                         perplexity=30, max_samples=None, figsize=(12, 10)):
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用自定义特征提取方法
        embeddings, labels, label_names, image_paths = self.extract_features_custom(
            dataset_path, max_samples=max_samples
        )
        
        embeddings_tsne = self.compute_tsne(embeddings, perplexity=perplexity)
        
        dataset_name = os.path.basename(dataset_path)
        title = f"t-SNE Visualization - {dataset_name}"
        save_path = os.path.join(output_dir, f"tsne_{dataset_name}.png")
        
        success = self.plot_tsne(
            embeddings_tsne, labels, label_names, title, save_path, figsize
        )
        
        if success:
            data_save_path = os.path.join(output_dir, f"tsne_data_{dataset_name}.npz")
            np.savez(
                data_save_path, 
                embeddings_tsne=embeddings_tsne, 
                labels=labels,
                label_names=label_names,
                image_paths=image_paths if image_paths is not None else []
            )
            logger.info(f"Feature data saved to: {data_save_path}")
            
            self._save_statistics(embeddings_tsne, labels, label_names, output_dir, dataset_name)
        
        return success
    
    def _save_statistics(self, embeddings_tsne, labels, label_names, output_dir, dataset_name):
        stats_path = os.path.join(output_dir, f"tsne_stats_{dataset_name}.txt")
        
        with open(stats_path, 'w') as f:
            f.write("t-SNE Visualization Statistics\n")
            f.write("=" * 40 + "\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Total samples: {len(embeddings_tsne)}\n")
            f.write(f"Number of classes: {len(np.unique(labels))}\n")
            f.write(f"t-SNE shape: {embeddings_tsne.shape}\n")
            f.write("\nClass distribution:\n")
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                # 找到该标签对应的名称
                label_indices = np.where(labels == label)[0]
                if len(label_indices) > 0 and len(label_names) > 0:
                    label_name = str(label_names[label_indices[0]])
                else:
                    label_name = f"Class {label}"
                f.write(f"  {label_name}: {count} samples\n")
        
        logger.info(f"Statistics saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize model features using t-SNE')
    parser.add_argument('--config', type=str, default='configs/train_megadescriptor.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for computation')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return 1
    
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        return 1
    
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset directory not found: {args.dataset}")
        return 1
    
    try:
        visualizer = TSNEVisualizer(args.config, args.checkpoint, args.device)
        success = visualizer.visualize_dataset(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            perplexity=args.perplexity,
            max_samples=args.max_samples
        )
        
        if success:
            logger.info("t-SNE visualization completed successfully")
            return 0
        else:
            logger.error("t-SNE visualization failed")
            return 1
            
    except Exception as e:
        logger.error(f"Error during t-SNE visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())