import logging
import os
from datetime import datetime

def setup_logging(log_dir, level=logging.INFO):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件路径
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # 配置日志
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir):
        self.logger = setup_logging(log_dir)
        self.metrics = {}
    
    def log_metrics(self, epoch, metrics_dict):
        """记录指标"""
        self.metrics[epoch] = metrics_dict
        self.logger.info(f"Epoch {epoch}: {metrics_dict}")
    
    def save_metrics(self, filepath):
        """保存指标到文件"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)