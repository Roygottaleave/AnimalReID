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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import MyAnimalDataset
from src.data.transforms import get_train_transforms, get_val_transforms

# 可选：保留随机三元组数据集，以便 mining=random 时使用
class TripletDatasetRandom(torch.utils.data.Dataset):
    def __init__(self, df, root, transform=None, col_path='path', col_label='identity'):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.col_path = col_path
        self.col_label = col_label

        # label -> indices
        self.label_to_indices = {}
        for idx, label in enumerate(self.df[self.col_label]):
            self.label_to_indices.setdefault(label, []).append(idx)
        self.labels = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        import random
        anchor_info = self.df.iloc[index]
        anchor_label = anchor_info[self.col_label]
        anchor_path = os.path.join(self.root, anchor_info[self.col_path])

        # positive index different from anchor
        pos_idx = index
        while pos_idx == index:
            pos_idx = random.choice(self.label_to_indices[anchor_label])
        pos_info = self.df.iloc[pos_idx]
        pos_path = os.path.join(self.root, pos_info[self.col_path])

        # negative: choose any different label
        neg_label = random.choice([l for l in self.labels if l != anchor_label])
        neg_idx = random.choice(self.label_to_indices[neg_label])
        neg_info = self.df.iloc[neg_idx]
        neg_path = os.path.join(self.root, neg_info[self.col_path])

        from PIL import Image
        a = Image.open(anchor_path).convert("RGB")
        p = Image.open(pos_path).convert("RGB")
        n = Image.open(neg_path).convert("RGB")

        if self.transform:
            a = self.transform(a)
            p = self.transform(p)
            n = self.transform(n)

        return a, p, n, anchor_label


class MegaDescriptorFinetuner:
    """MegaDescriptor 微调器（支持 ArcFace 和 Triplet；Triplet 支持 semi-hard mining）"""
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self._setup_directories()
        set_seed(self.config['project']['seed'])

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_directories(self):
        os.makedirs(self.config['output']['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['output']['log_dir'], exist_ok=True)
        os.makedirs("data/splits", exist_ok=True)

    def _save_checkpoint(self, epoch, metric_value):
        checkpoint = {
            'epoch': epoch,
            'backbone_state_dict': self.backbone.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metric_value': metric_value,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        checkpoint_path = os.path.join(self.config['output']['checkpoint_dir'], f'checkpoint_epoch_{epoch:03d}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def prepare_data(self):
        print("Preparing data...")
        self.custom_dataset = MyAnimalDataset(self.config['data']['root_dir'])
        self.custom_dataset.analyze_dataset()

        # 使用 ClosedSetSplit 先划分训练+验证，再划分最终测试集
        splitter = splits.ClosedSetSplit(ratio_train=self.config['data']['split_ratio'])
        idx_trainval, idx_test = splitter.split(self.custom_dataset.df)[0]  # train+val, test
        train_val_df = self.custom_dataset.df.loc[idx_trainval].reset_index(drop=True)
        test_df = self.custom_dataset.df.loc[idx_test].reset_index(drop=True)

        # 再对 train_val_df 做 train/val 划分
        val_ratio = self.config['data'].get('val_ratio', 0.2)
        num_val = int(len(train_val_df) * val_ratio)
        self.df_val = train_val_df.sample(n=num_val, random_state=self.config['project']['seed']).reset_index(drop=True)
        self.df_train = train_val_df.drop(self.df_val.index).reset_index(drop=True)
        self.df_test = test_df  # 测试集仅用于最终评估

        train_transform = get_train_transforms(self.config)
        val_transform = get_val_transforms(self.config)
        test_transform = get_val_transforms(self.config)

        loss_name = self.config['training']['loss']['name'].lower()
        triplet_mining = self.config['training']['loss'].get('triplet', {}).get('mining', 'random')

        if loss_name == "triplet" and triplet_mining == "random":
            self.train_dataset = TripletDatasetRandom(
                df=self.df_train,
                root=self.config['data']['root_dir'],
                transform=train_transform
            )
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['data']['batch_size'],
                                           shuffle=True, num_workers=self.config['data']['num_workers'])
        else:
            self.train_dataset = ImageDataset(
                metadata=self.df_train,
                root=self.config['data']['root_dir'],
                transform=train_transform,
                col_path='path',
                col_label='identity'
            )
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['data']['batch_size'],
                                           shuffle=True, num_workers=self.config['data']['num_workers'])

        self.val_dataset = ImageDataset(
            metadata=self.df_val,
            root=self.config['data']['root_dir'],
            transform=val_transform,
            col_path='path',
            col_label='identity'
        )
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config['data']['batch_size'],
                                     shuffle=False, num_workers=self.config['data']['num_workers'])

        self.test_dataset = ImageDataset(
            metadata=self.df_test,
            root=self.config['data']['root_dir'],
            transform=test_transform,
            col_path='path',
            col_label='identity'
        )
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config['data']['batch_size'],
                                      shuffle=False, num_workers=self.config['data']['num_workers'])

        print("Data preparation completed")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Testing samples: {len(self.test_dataset)}")

        return len(self.custom_dataset.df['identity'].unique())

    def setup_model(self, num_classes):
        print("Initializing model...")
        self.backbone = timm.create_model(self.config['model']['name'], num_classes=0,
                                          pretrained=self.config['model'].get('pretrained', True))
        self.backbone.to(self.device)

        if self.config['model'].get('freeze_backbone', False):
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("Backbone frozen: True")
        else:
            print("Backbone frozen: False")

        loss_cfg = self.config['training']['loss']
        loss_name = loss_cfg['name'].lower()

        if loss_name == 'arcface':
            arc_cfg = loss_cfg.get('arcface', {})
            scale = float(arc_cfg.get('scale', 64.0))
            margin = float(arc_cfg.get('margin', 0.5))
            self.objective = ArcFaceLoss(num_classes=num_classes,
                                         embedding_size=int(self.config['training']['loss'].get('embedding_size', 768)),
                                         margin=margin, scale=scale)
            print("Using ArcFaceLoss")
            params = itertools.chain(self.backbone.parameters(), self.objective.parameters())
        elif loss_name == 'triplet':
            trip_cfg = loss_cfg.get('triplet', {})
            margin = float(trip_cfg.get('margin', 0.3))
            self.objective = nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
            print(f"Using TripletMarginLoss with margin={margin}")
            params = self.backbone.parameters()
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

        lr = float(self.config['training']['learning_rate'])
        weight_decay = float(self.config['training']['weight_decay'])

        if self.config['training']['optimizer'].lower() == 'adamw':
            self.optimizer = AdamW(params=params, lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = SGD(params=params, lr=lr, momentum=0.9, weight_decay=weight_decay)

        if self.config['training']['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config['training']['epochs'])
        else:
            self.scheduler = None

        print("Model initialization completed")

    def compute_metrics(self, dataset, name="Dataset"):
        """
        Compute evaluation metrics: mAP, CMC (rank-1,5,10).
        Includes safe normalization and debug printing to avoid overflow / NaN issues.
        """
        self.backbone.eval()
        feature_extractor = DeepFeatures(model=self.backbone,
                                         batch_size=self.config['data']['batch_size'],
                                         num_workers=self.config['data']['num_workers'],
                                         device=str(self.device))
        print(f"Extracting features for {name}...")
        features = feature_extractor(dataset)
        emb = features.features

        # 防止数值过大 / nan / inf
        emb = np.nan_to_num(emb, nan=0.0, posinf=1e3, neginf=-1e3)

        # L2 normalize safely
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        features_array = emb / norms

        labels = dataset.labels
        
        # 添加标签调试信息
        print(f"[DEBUG] Label mapping:")
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            print(f"  Label {label}: {len(label_indices)} samples")
        
        # 如果是数字标签，尝试获取原始标签名
        if hasattr(dataset, 'df') and 'identity' in dataset.df.columns:
            original_labels = dataset.df['identity'].values
            print(f"[DEBUG] Original label names: {np.unique(original_labels)}")       

        # 构建 query/gallery
        query_indices, gallery_indices = [], []
        identity_to_indices = {}
        for idx, label in enumerate(labels):
            identity_to_indices.setdefault(label, []).append(idx)
        for inds in identity_to_indices.values():
            query_indices.append(inds[0])
            gallery_indices.extend(inds[1:])
        
        if len(gallery_indices) == 0:
            gallery_indices = query_indices.copy()

        query_feats = features_array[query_indices]
        query_labels = [labels[i] for i in query_indices]
        gallery_feats = features_array[gallery_indices]
        gallery_labels = [labels[i] for i in gallery_indices]

        # 稳定的余弦相似度计算
        def stable_cosine_similarity(a, b):
            epsilon = 1e-10
            a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + epsilon)
            b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + epsilon)
            similarity = np.dot(a_norm, b_norm.T)
            similarity = np.clip(similarity, -1.0, 1.0)
            return similarity

        similarity_matrix = stable_cosine_similarity(query_feats, gallery_feats)

        # mAP 计算
        average_precisions = []
        for i, q_label in enumerate(query_labels):
            sorted_indices = np.argsort(similarity_matrix[i])[::-1]
            relevant = 0
            precision_at_k = []
            for k, idx in enumerate(sorted_indices):
                if gallery_labels[idx] == q_label:
                    relevant += 1
                    precision_at_k.append(relevant / (k + 1))
            ap = np.mean(precision_at_k) if relevant > 0 else 0
            average_precisions.append(ap)
        mean_ap = np.mean(average_precisions)

        # CMC 计算
        ranks = [1, 5, 10]
        cmc_scores = {}
        for rank in ranks:
            correct = 0
            for i, q_label in enumerate(query_labels):
                sorted_indices = np.argsort(similarity_matrix[i])[::-1]
                if any(gallery_labels[idx] == q_label for idx in sorted_indices[:rank]):
                    correct += 1
            cmc_scores[rank] = correct / len(query_labels)

        self.backbone.train()

        # 在返回前添加详细分析
        print(f"[ANALYSIS] {name} - Query: {len(query_labels)}, Gallery: {len(gallery_labels)}")
        print(f"[ANALYSIS] Unique query labels: {len(set(query_labels))}")
        print(f"[ANALYSIS] Unique gallery labels: {len(set(gallery_labels))}")
        
        # 分析相似度矩阵
        print(f"[ANALYSIS] Similarity matrix range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
        
        # 检查每个query的最佳匹配
        for i, q_label in enumerate(query_labels[:3]):  # 只看前3个
            best_match_idx = np.argmax(similarity_matrix[i])
            best_match_label = gallery_labels[best_match_idx]
            best_similarity = similarity_matrix[i, best_match_idx]
            print(f"[ANALYSIS] Query {i} (label {q_label}) -> Best match: {best_match_label}, similarity: {best_similarity:.3f}")

        return {'mAP': mean_ap, 'rank1': cmc_scores[1], 'rank5': cmc_scores[5], 'rank10': cmc_scores[10]}

    def _embeddings_from_batch(self, images):
        self.backbone.eval()
        with torch.no_grad():
            images = images.to(self.device)
            emb = self.backbone(images)
            if isinstance(emb, tuple):
                emb = emb[0]
            emb = emb.detach()
        return emb.cpu()

    def _batch_semi_hard_triplets(self, embeddings, labels, margin):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        emb_np = embeddings.numpy()
        B = emb_np.shape[0]
        dists = np.linalg.norm(emb_np[:, None, :] - emb_np[None, :, :], axis=2)
        triplets = []
        for a in range(B):
            a_label = labels[a]
            pos_indices = np.where(labels == a_label)[0]
            neg_indices = np.where(labels != a_label)[0]
            pos_indices = pos_indices[pos_indices != a]
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
            p = np.random.choice(pos_indices)
            dist_ap = dists[a, p]
            cand_neg = [n for n in neg_indices if dists[a, n] > dist_ap and dists[a, n] < dist_ap + margin]
            if len(cand_neg) > 0:
                n = np.random.choice(cand_neg)
            else:
                n = neg_indices[np.argmin(dists[a, neg_indices])]
            triplets.append((a, p, n))
        return triplets

    def create_trainer(self):
        loss_name = self.config['training']['loss']['name'].lower()
        val_freq = self.config['output'].get('val_frequency', 5)
        self.best_mAP = 0.0
        self.current_epoch = 0  # 添加当前epoch计数器

        def epoch_callback(trainer, epoch_data):
            self.current_epoch += 1
            train_loss = epoch_data.get('train_loss_epoch_avg', 0.0)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 只在val_frequency指定的epoch进行验证
            if self.current_epoch % val_freq == 0 or self.current_epoch == self.config['training']['epochs']:
                validation_metrics = self.compute_metrics(self.val_dataset, name="Validation Set")
                if validation_metrics['mAP'] > self.best_mAP:
                    self.best_mAP = validation_metrics['mAP']
                print(f"Epoch {self.current_epoch:3d}/{self.config['training']['epochs']} | "
                      f"LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | "
                      f"Val-mAP: {validation_metrics['mAP']:.4f} | "
                      f"Rank-1: {validation_metrics['rank1']:.4f} | "
                      f"Rank-5: {validation_metrics['rank5']:.4f}")
                
                if self.current_epoch % self.config['output']['save_frequency'] == 0 or self.current_epoch == self.config['training']['epochs']:
                    self._save_checkpoint(self.current_epoch, validation_metrics['mAP'])
            else:
                # 不验证时只打印训练信息
                print(f"Epoch {self.current_epoch:3d}/{self.config['training']['epochs']} | "
                      f"LR: {current_lr:.2e} | Train Loss: {train_loss:.4f}")

        if self.config['training']['loss']['name'].lower() == 'arcface':
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
        else:
            self.trainer = None
        return self.trainer

    def train(self):
        print("=" * 60)
        print("MegaDescriptor Fine-tuning Training")
        print("=" * 60)

        num_classes = self.prepare_data()
        self.setup_model(num_classes)
        trainer = self.create_trainer()

        loss_name = self.config['training']['loss']['name'].lower()
        trip_cfg = self.config['training']['loss'].get('triplet', {})
        triplet_mining = trip_cfg.get('mining', 'random').lower()

        # --- ArcFace training ---
        if loss_name == 'arcface':
            print("Starting ArcFace training (using BasicTrainer)...")
            print(f"Device: {self.device}")
            trainer.train()

        # --- Triplet random ---
        elif loss_name == 'triplet' and triplet_mining == 'random':
            print("Starting Triplet (random sampling) training...")
            epochs = self.config['training']['epochs']
            margin = float(trip_cfg.get('margin', 0.3))
            for epoch in range(1, epochs + 1):
                self.backbone.train()
                running_loss = 0.0
                count = 0
                for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}"):
                    a, p, n, labels = batch
                    a, p, n = a.to(self.device), p.to(self.device), n.to(self.device)
                    emb_a = self.backbone(a)
                    emb_p = self.backbone(p)
                    emb_n = self.backbone(n)
                    loss = self.objective(emb_a, emb_p, emb_n)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                    count += 1
                if self.scheduler:
                    self.scheduler.step()
                avg_loss = running_loss / max(1, count)
                validation_metrics = self.compute_metrics(self.val_dataset, name="Validation Set")
                if validation_metrics['mAP'] > self.best_mAP:
                    self.best_mAP = validation_metrics['mAP']
                print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} | "
                      f"Val-mAP: {validation_metrics['mAP']:.4f} | Rank-1: {validation_metrics['rank1']:.4f}")
                if epoch % self.config['output']['save_frequency'] == 0 or epoch == epochs:
                    self._save_checkpoint(epoch, validation_metrics['mAP'])

        # --- Triplet semi-hard ---
        elif loss_name == 'triplet' and triplet_mining == 'semi-hard':
            print("Starting Triplet training with semi-hard mining (batch-based)...")
            epochs = self.config['training']['epochs']
            margin = float(trip_cfg.get('margin', 0.3))
            val_freq = self.config['output'].get('val_frequency', 5)
            for epoch in range(1, epochs + 1):
                self.backbone.train()
                running_loss = 0.0
                count = 0
                for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}"):
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)
                    emb = self.backbone(images)
                    if isinstance(emb, tuple):
                        emb = emb[0]
                    emb_cpu = emb.detach().cpu()
                    labels_cpu = labels.detach().cpu()
                    triplets = self._batch_semi_hard_triplets(emb_cpu, labels_cpu, margin)
                    if len(triplets) == 0:
                        continue
                    idx_a = torch.tensor([t[0] for t in triplets], dtype=torch.long, device=self.device)
                    idx_p = torch.tensor([t[1] for t in triplets], dtype=torch.long, device=self.device)
                    idx_n = torch.tensor([t[2] for t in triplets], dtype=torch.long, device=self.device)
                    emb_a = emb[idx_a]
                    emb_p = emb[idx_p]
                    emb_n = emb[idx_n]
                    loss = self.objective(emb_a, emb_p, emb_n)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                    count += 1
                if self.scheduler:
                    self.scheduler.step()
                avg_loss = running_loss / max(1, count)
                validation_metrics = self.compute_metrics(self.val_dataset, name="Validation Set")
                if validation_metrics['mAP'] > self.best_mAP:
                    self.best_mAP = validation_metrics['mAP']
                print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} | "
                      f"Val-mAP: {validation_metrics['mAP']:.4f} | Rank-1: {validation_metrics['rank1']:.4f}")
                if epoch % self.config['output']['save_frequency'] == 0 or epoch == epochs:
                    self._save_checkpoint(epoch, validation_metrics['mAP'])
        else:
            raise ValueError("Unsupported training configuration.")

        # --- 最终在测试集上评估 ---
        print("\nFinal Evaluation on Testing Set")
        final_metrics = self.compute_metrics(self.test_dataset, name="Testing Set")
        print(final_metrics)
        final_path = os.path.join(self.config['output']['checkpoint_dir'], 'final_model.pth')
        torch.save({'backbone_state_dict': self.backbone.state_dict(), 'final_metrics': final_metrics}, final_path)
        print(f"Final model saved: {final_path}")
        return final_path