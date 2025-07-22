"""
配置文件：所有超参数和路径设置
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    """模型相关配置"""
    base_model_path: str = "/path/to/llama-7b"
    tallrec_adapter_path: str = "/path/to/tallrec-adapter"
    item_embedding_path: str = "./data/item_embedding.pt"
    
    # 模型参数
    max_length: int = 256
    batch_size: int = 32
    temperature: float = 0.7
    load_in_8bit: bool = True

@dataclass
class DataConfig:
    """数据相关配置"""
    item_names_file: str = "./data/item_names.txt"
    train_data_path: str = "./data/train.json"
    test_data_path: str = "./data/test.json"
    
    # 数据处理参数
    max_history_length: int = 10
    min_history_length: int = 1

@dataclass
class FusionConfig:
    """融合相关配置"""
    fusion_method: str = "adaptive"  # linear, weighted, adaptive, ensemble
    optimal_alpha: float = 0.6
    decision_threshold: float = 0.5
    confidence_threshold: float = 0.3
    
    # 权重搜索范围
    alpha_range: List[float] = None
    threshold_range: List[float] = None
    
    def __post_init__(self):
        if self.alpha_range is None:
            self.alpha_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if self.threshold_range is None:
            self.threshold_range = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

@dataclass
class TrainingConfig:
    """训练相关配置"""
    seed: int = 42
    device: str = "auto"
    num_workers: int = 4
    
    # 评估参数
    eval_batch_size: int = 32
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1", "auc"]

class Config:
    """主配置类"""
    def __init__(self, config_dict: Dict[str, Any] = None):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.fusion = FusionConfig()
        self.training = TrainingConfig()
        
        # 如果提供了配置字典，更新配置
        if config_dict:
            self.update_from_dict(config_dict)
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新配置"""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "fusion": self.fusion.__dict__,
            "training": self.training.__dict__
        }
    
    def save(self, filepath: str):
        """保存配置到文件"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str):
        """从文件加载配置"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(config_dict)

# 默认配置实例
default_config = Config()