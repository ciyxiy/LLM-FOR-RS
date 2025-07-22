"""
TALLRec增强编码器：使用TALLRec模型提取增强的用户偏好嵌入
"""

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from config import Config
from utils import setup_logging, batch_data, load_json, save_json
import os
logger = setup_logging()

class TALLRecEnhancedEncoder:
    def __init__(self, config: Config):
        """
        初始化TALLRec增强编码器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """加载TALLRec增强模型"""
        logger.info("Loading TALLRec enhanced model...")
        
        # 加载基础LLaMA模型
        logger.info(f"Loading base model from: {self.config.model.base_model_path}")
        self.tokenizer = LlamaTokenizer.from_pretrained(self.config.model.base_model_path)
        self.base_model = LlamaForCausalLM.from_pretrained(
            self.config.model.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=self.config.model.load_in_8bit
        )
        
        # 加载TALLRec适配器
        tallrec_adapter_path = self.config.model.tallrec_adapter_path
        if tallrec_adapter_path and os.path.exists(tallrec_adapter_path):
            logger.info(f"Loading TALLRec adapter from: {tallrec_adapter_path}")
            self.model = PeftModel.from_pretrained(self.base_model, tallrec_adapter_path)
        else:
            logger.warning("TALLRec adapter not found, using base model")
            self.model = self.base_model
        
        # 设置模型配置
        self.setup_model_config()
        
        logger.info("TALLRec enhanced model loaded successfully!")
    
    def setup_model_config(self):
        """设置模型配置"""
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.model.eval()
        self.tokenizer.padding_side = "left"
    
    def generate_enhanced_preference_prompt(self, user_history: List[str], target_item: str) -> str:
        """
        生成增强的用户偏好理解提示
        
        Args:
            user_history: 用户历史
            target_item: 目标物品
        
        Returns:
            enhanced_prompt: 增强提示
        """
        if user_history:
            history_str = ", ".join(user_history[-5:])  # 取最近5个
            prompt = f"""### Instruction:
Based on the user's interaction history, analyze their detailed preferences and explain their likely interests regarding the target item.

### Input:
User History: {history_str}
Target Item: {target_item}

### Response:
This user demonstrates a strong preference for"""
        else:
            prompt = f"""### Instruction:
Analyze the characteristics and appeal of the following item for potential users.

### Input:
Target Item: {target_item}

### Response:
This item would appeal to users who"""
        
        return prompt
    
    def extract_enhanced_embeddings(self, prompts: List[str]) -> torch.Tensor:
        """
        提取增强的嵌入表示
        
        Args:
            prompts: 提示列表
        
        Returns:
            enhanced_embeddings: 增强嵌入张量 [num_prompts, embedding_dim]
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Extracting enhanced embeddings for {len(prompts)} prompts")
        
        embeddings = []
        
        for batch_prompts in tqdm(
            batch_data(prompts, self.config.model.batch_size), 
            desc="Extracting enhanced embeddings"
        ):
            # 分词和编码
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.model.max_length
            )
            
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            with torch.no_grad():
                # 使用TALLRec增强模型进行编码
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                hidden_states = outputs.hidden_states
                
                # 使用多层融合获得更好的表示
                if len(hidden_states) >= 3:
                    # 融合最后3层
                    last_3_layers = torch.stack([
                        hidden_states[-1], hidden_states[-2], hidden_states[-3]
                    ])
                    fused_hidden = last_3_layers.mean(dim=0)  # [batch, seq_len, hidden_dim]
                else:
                    fused_hidden = hidden_states[-1]
                
                # 使用注意力加权平均而不是简单的最后一个token
                # 计算有效长度的注意力权重
                attention_weights = attention_mask.float()
                attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True)
                
                # 加权平均池化
                weighted_embeddings = (fused_hidden * attention_weights.unsqueeze(-1)).sum(dim=1)
                
                embeddings.append(weighted_embeddings.cpu())
        
        enhanced_embeddings = torch.cat(embeddings, dim=0)
        
        logger.info(f"Generated enhanced embeddings shape: {enhanced_embeddings.shape}")
        
        return enhanced_embeddings
    
    def encode_user_preferences(self, data: List[Dict]) -> Tuple[torch.Tensor, List[str]]:
        """
        编码用户偏好为增强嵌入
        
        Args:
            data: 包含用户历史和目标物品的数据
        
        Returns:
            user_embeddings: 用户嵌入张量
            enhanced_prompts: 增强提示列表
        """
        logger.info(f"Encoding user preferences for {len(data)} samples")
        
        # 生成增强提示
        enhanced_prompts = []
        for sample in data:
            prompt = self.generate_enhanced_preference_prompt(
                sample['user_history'],
                sample['item_title']
            )
            enhanced_prompts.append(prompt)
        
        # 提取增强嵌入
        user_embeddings = self.extract_enhanced_embeddings(enhanced_prompts)
        
        return user_embeddings, enhanced_prompts
    
    def analyze_embedding_quality(self, embeddings: torch.Tensor, 
                                 labels: List[int]) -> Dict[str, float]:
        """
        分析嵌入质量
        
        Args:
            embeddings: 嵌入张量
            labels: 对应标签
        
        Returns:
            quality_metrics: 质量指标
        """
        logger.info("Analyzing embedding quality...")
        
        # 转换为numpy数组
        embeddings_np = embeddings.numpy()
        labels_np = np.array(labels)
        
        # 计算正负样本的嵌入统计
        positive_embeddings = embeddings_np[labels_np == 1]
        negative_embeddings = embeddings_np[labels_np == 0]
        
        quality_metrics = {}
        
        if len(positive_embeddings) > 0 and len(negative_embeddings) > 0:
            # 计算类内距离和类间距离
            pos_centroid = np.mean(positive_embeddings, axis=0)
            neg_centroid = np.mean(negative_embeddings, axis=0)
            
            # 类间距离
            inter_class_distance = np.linalg.norm(pos_centroid - neg_centroid)
            
            # 类内距离（平均）
            pos_intra_distances = [
                np.linalg.norm(emb - pos_centroid) for emb in positive_embeddings
            ]
            neg_intra_distances = [
                np.linalg.norm(emb - neg_centroid) for emb in negative_embeddings
            ]
            
            avg_pos_intra_distance = np.mean(pos_intra_distances)
            avg_neg_intra_distance = np.mean(neg_intra_distances)
            avg_intra_distance = (avg_pos_intra_distance + avg_neg_intra_distance) / 2
            
            # 分离度（类间距离/类内距离）
            separability = inter_class_distance / (avg_intra_distance + 1e-8)
            
            quality_metrics.update({
                'inter_class_distance': float(inter_class_distance),
                'avg_intra_distance': float(avg_intra_distance),
                'separability': float(separability),
                'positive_samples': len(positive_embeddings),
                'negative_samples': len(negative_embeddings)
            })
        
        # 嵌入的基本统计
        quality_metrics.update({
            'embedding_dim': embeddings.shape[1],
            'total_samples': embeddings.shape[0],
            'mean_norm': float(np.mean(np.linalg.norm(embeddings_np, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings_np, axis=1))),
            'mean_activation': float(np.mean(embeddings_np)),
            'std_activation': float(np.std(embeddings_np))
        })
        
        return quality_metrics
    
    def save_embeddings(self, embeddings: torch.Tensor, prompts: List[str], 
                       quality_metrics: Dict, output_path: str):
        """
        保存嵌入和相关信息
        
        Args:
            embeddings: 嵌入张量
            prompts: 提示列表
            quality_metrics: 质量指标
            output_path: 输出路径
        """
        logger.info(f"Saving enhanced embeddings to: {output_path}")
        
        # 保存嵌入张量
        torch.save(embeddings, output_path)
        
        # 保存元数据
        metadata_path = output_path.replace('.pt', '_metadata.json')
        metadata = {
            'embedding_shape': list(embeddings.shape),
            'embedding_dtype': str(embeddings.dtype),
            'model_config': {
                'base_model_path': self.config.model.base_model_path,
                'tallrec_adapter_path': self.config.model.tallrec_adapter_path,
                'max_length': self.config.model.max_length,
                'batch_size': self.config.model.batch_size
            },
            'quality_metrics': quality_metrics,
            'sample_prompts': prompts[:5]  # 保存前5个示例
        }
        
        save_json(metadata, metadata_path)
        
        logger.info(f"Enhanced embeddings and metadata saved")
        logger.info(f"Embedding shape: {embeddings.shape}")
        logger.info(f"Quality metrics: {quality_metrics}")
    
    def run(self, data: List[Dict], output_path: str = None) -> Tuple[torch.Tensor, Dict]:
        """
        运行完整的增强编码流程
        
        Args:
            data: 输入数据
            output_path: 输出路径（可选）
        
        Returns:
            user_embeddings: 用户嵌入
            quality_metrics: 质量指标
        """
        logger.info("=== TALLRec Enhanced Encoding Started ===")
        
        try:
            # 1. 加载模型
            self.load_model()
            
            # 2. 编码用户偏好
            user_embeddings, enhanced_prompts = self.encode_user_preferences(data)
            
            # 3. 分析嵌入质量
            labels = [sample['label'] for sample in data]
            quality_metrics = self.analyze_embedding_quality(user_embeddings, labels)
            
            # 4. 保存结果（如果指定了输出路径）
            if output_path:
                self.save_embeddings(user_embeddings, enhanced_prompts, quality_metrics, output_path)
            
            logger.info("=== TALLRec Enhanced Encoding Completed ===")
            
            return user_embeddings, quality_metrics
            
        except Exception as e:
            logger.error(f"Error during enhanced encoding: {str(e)}")
            raise

def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="TALLRec Enhanced User Preference Encoding")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--data_file", type=str, required=True, help="Input data file")
    parser.add_argument("--output_path", type=str, help="Output embeddings path")
    
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        config = Config.load(args.config)
    else:
        config = Config()
        logger.warning(f"Config file not found: {args.config}, using default config")
    
    # 检查输入文件
    if not os.path.exists(args.data_file):
        logger.error(f"Data file not found: {args.data_file}")
        return
    
    # 运行编码器
    try:
        # 加载数据
        data = load_json(args.data_file)
        logger.info(f"Loaded {len(data)} samples")
        
        # 初始化编码器
        encoder = TALLRecEnhancedEncoder(config)
        
        # 运行编码
        user_embeddings, quality_metrics = encoder.run(data, args.output_path)
        
        logger.info(f"✅ Success! Generated enhanced embeddings:")
        logger.info(f"  Shape: {user_embeddings.shape}")
        logger.info(f"  Separability: {quality_metrics.get('separability', 'N/A'):.4f}")
        
    except Exception as e:
        logger.error(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()