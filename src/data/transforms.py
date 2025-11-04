import torchvision.transforms as T

def get_train_transforms(config):
    """获取训练时的数据增强变换"""
    aug_config = config['data']['augmentation']['train']
    
    return T.Compose([
        T.Resize(aug_config['resize']),
        T.RandomCrop(aug_config['random_crop']),
        T.RandomHorizontalFlip(p=aug_config['random_horizontal_flip']),
        T.ColorJitter(
            brightness=aug_config['color_jitter'][0],
            contrast=aug_config['color_jitter'][1],
            saturation=aug_config['color_jitter'][2],
            hue=aug_config['color_jitter'][3]
        ),
        T.RandomRotation(degrees=aug_config['random_rotation']),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_val_transforms(config):
    """获取验证时的数据变换"""
    aug_config = config['data']['augmentation']['val']
    
    return T.Compose([
        T.Resize(aug_config['resize']),
        T.CenterCrop(aug_config['center_crop']),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])