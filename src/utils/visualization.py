import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE
import torch

def plot_training_metrics(metrics_dict, save_path=None):
    """绘制训练指标"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = list(metrics_dict.keys())
    
    # 训练损失和准确率
    train_losses = [metrics_dict[epoch]['train_loss'] for epoch in epochs]
    train_accs = [metrics_dict[epoch]['train_accuracy'] for epoch in epochs]
    val_losses = [metrics_dict[epoch]['val_loss'] for epoch in epochs]
    val_accs = [metrics_dict[epoch]['val_accuracy'] for epoch in epochs]
    
    # 损失曲线
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 准确率曲线
    axes[0, 1].plot(epochs, train_accs, 'b-', label='Train Accuracy')
    axes[0, 1].plot(epochs, val_accs, 'r-', label='Val Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 学习率曲线（如果有）
    if 'learning_rate' in metrics_dict[epochs[0]]:
        learning_rates = [metrics_dict[epoch]['learning_rate'] for epoch in epochs]
        axes[1, 0].plot(epochs, learning_rates, 'g-')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved: {save_path}")
    
    plt.show()

def visualize_embeddings(features, labels, save_path=None):
    """可视化特征嵌入"""
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 8))
    
    # 创建散点图
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.7, s=10)
    
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Feature Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature visualization saved: {save_path}")
    
    plt.show()