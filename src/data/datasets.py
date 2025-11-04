import pandas as pd
import os
from wildlife_datasets.datasets import WildlifeDataset

class MyAnimalDataset(WildlifeDataset):
    """
    处理动物数据集结构
    目录结构:
    dataset_root/
        ├── AI/           # 个体ID: AI
        │   ├── AI_2018.03.19.sep.DSC_0061.NEF
        │   └── ...
        ├── AVV/          # 个体ID: AVV  
        │   ├── AV_2018-02-13.sep.DSC_0203.NEF
        │   └── ...
        └── ...
    """
    
    def create_catalogue(self) -> pd.DataFrame:
        """从文件夹结构创建数据集目录"""
        data = []
        image_counter = 1
        
        print(f"Scanning directory: {self.root}")
        
        # 遍历根目录下的所有项目
        for item in os.listdir(self.root):
            item_path = os.path.join(self.root, item)
            
            # 只处理目录（每个目录对应一个个体）
            if os.path.isdir(item_path):
                individual_id = item
                print(f"Processing individual: {individual_id}")
                
                # 遍历该个体目录下的所有图像文件
                for filename in os.listdir(item_path):
                    # 支持多种图像格式
                    if filename.lower().endswith(('.nef', '.jpg', '.jpeg', '.png', '.tiff')):
                        # 构建相对路径
                        relative_path = os.path.join(individual_id, filename)
                        
                        record = {
                            'image_id': f"{individual_id}_{image_counter:06d}",
                            'identity': individual_id,
                            'path': relative_path,
                            'filename': filename,
                            'individual': individual_id
                        }
                        data.append(record)
                        image_counter += 1
        
        if not data:
            raise ValueError(f"No image files found in directory: {self.root}")
        
        df = pd.DataFrame(data)
        print(f"Successfully loaded {len(df)} images from {df['identity'].nunique()} individuals")
        
        return df

    def analyze_dataset(self):
        """分析数据集统计信息"""
        df = self.df
        
        print("=" * 50)
        print("Dataset Analysis Report")
        print("=" * 50)
        print(f"Total images: {len(df)}")
        print(f"Number of individuals: {df['identity'].nunique()}")
        
        # 每个个体的图片数量统计
        individual_counts = df['identity'].value_counts()
        print(f"Images per individual statistics:")
        print(f"  Maximum: {individual_counts.max()}")
        print(f"  Minimum: {individual_counts.min()}")
        print(f"  Average: {individual_counts.mean():.1f}")
        print(f"  Median: {individual_counts.median()}")
        
        # 显示前几个个体
        print(f"Individual distribution (top 10):")
        for individual, count in individual_counts.head(10).items():
            print(f"  {individual}: {count} images")
        
        # 文件格式统计
        file_extensions = df['filename'].apply(lambda x: os.path.splitext(x)[1].lower()).value_counts()
        print(f"File format distribution:")
        for ext, count in file_extensions.items():
            print(f"  {ext}: {count} files")
        
        return df