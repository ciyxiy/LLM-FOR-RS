
import os
import json
import torch
import random
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def setup_logging(log_file: str = "tallrec_bigrec.log", level: str = "INFO"):
    """设置日志"""
    log_level = getattr(logging, level.upper())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 设置deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_str: str = "auto"):
    """获取计算设备"""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_str)

def save_json(data: Any, filepath: str):
    """保存JSON文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath: str) -> Any:
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def batch_data(data: List, batch_size: int):
    """批处理数据生成器"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def normalize_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """标准化分数到[0,1]范围"""
    if method == "minmax":
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
        else:
            return np.ones_like(scores) * 0.5
    
    elif method == "sigmoid":
        return 1 / (1 + np.exp(-scores))
    
    elif method == "softmax":
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / np.sum(exp_scores)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def calculate_metrics(true_labels: List[int], predictions: List[int], 
                     scores: Optional[List[float]] = None) -> Dict[str, float]:
    """计算评估指标"""
    metrics = {}
    
    # 基础分类指标
    metrics['accuracy'] = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', zero_division=0
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # AUC指标（如果提供了分数）
    if scores is not None:
        try:
            metrics['auc'] = roc_auc_score(true_labels, scores)
        except ValueError:
            metrics['auc'] = 0.0
    
    return metrics

def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """打印格式化的指标"""
    print(f"\n=== {title} ===")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name.capitalize()}: {value:.4f}")
        else:
            print(f"{metric_name.capitalize()}: {value}")

def create_directory(path: str):
    """创建目录"""
    os.makedirs(path, exist_ok=True)

def check_file_exists(filepath: str) -> bool:
    """检查文件是否存在"""
    return os.path.exists(filepath)

def get_file_size(filepath: str) -> str:
    """获取文件大小（人类可读格式）"""
    size_bytes = os.path.getsize(filepath)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} TB"

class ProgressTracker:
    """进度跟踪器"""
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
    
    def update(self, step: int = 1):
        """更新进度"""
        self.current_step += step
        progress = self.current_step / self.total_steps * 100
        print(f"\r{self.description}: {progress:.1f}% ({self.current_step}/{self.total_steps})", end='')
        
        if self.current_step >= self.total_steps:
            print()  # 完成时换行

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        
    def __call__(self, score: float) -> bool:
        """检查是否应该早停"""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

def filter_valid_samples(data: List[Dict], item_embedding_size: int) -> List[Dict]:
    """过滤有效样本"""
    valid_samples = []
    invalid_count = 0
    
    for sample in data:
        item_id = sample.get('item_id', -1)
        if 0 <= item_id < item_embedding_size:
            valid_samples.append(sample)
        else:
            invalid_count += 1
    
    print(f"Filtered {len(valid_samples)} valid samples, {invalid_count} invalid samples")
    return valid_samples

def analyze_data_distribution(data: List[Dict]) -> Dict[str, Any]:
    """分析数据分布"""
    if not data:
        return {}
    
    # 标签分布
    labels = [sample.get('label', 0) for sample in data]
    label_counts = {0: labels.count(0), 1: labels.count(1)}
    
    # 用户历史长度分布
    history_lengths = [len(sample.get('user_history', [])) for sample in data]
    
    # 用户数量
    user_ids = set(sample.get('user_id', '') for sample in data)
    
    # 物品数量
    item_ids = set(sample.get('item_id', -1) for sample in data)
    
    analysis = {
        'total_samples': len(data),
        'label_distribution': label_counts,
        'label_ratio': label_counts[1] / (label_counts[0] + label_counts[1]) if (label_counts[0] + label_counts[1]) > 0 else 0,
        'unique_users': len(user_ids),
        'unique_items': len(item_ids),
        'avg_history_length': np.mean(history_lengths),
        'max_history_length': np.max(history_lengths) if history_lengths else 0,
        'min_history_length': np.min(history_lengths) if history_lengths else 0
    }
    
    return analysis

def print_data_analysis(analysis: Dict[str, Any], title: str = "Data Analysis"):
    """打印数据分析结果"""
    print(f"\n=== {title} ===")
    print(f"Total samples: {analysis.get('total_samples', 0)}")
    print(f"Label distribution: {analysis.get('label_distribution', {})}")
    print(f"Positive ratio: {analysis.get('label_ratio', 0):.3f}")
    print(f"Unique users: {analysis.get('unique_users', 0)}")
    print(f"Unique items: {analysis.get('unique_items', 0)}")
    print(f"History length - Avg: {analysis.get('avg_history_length', 0):.1f}, "
          f"Min: {analysis.get('min_history_length', 0)}, "
          f"Max: {analysis.get('max_history_length', 0)}")