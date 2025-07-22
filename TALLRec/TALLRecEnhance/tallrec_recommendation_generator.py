

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from config import Config
from utils import setup_logging, batch_data
import os

logger = setup_logging()

class TALLRecRecommendationGenerator:
    def __init__(self, config: Config):
       
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.tokenizer = None
        
        
        self.yes_token_id = 8241  # "Yes" token ID
        self.no_token_id = 3782   # "No" token ID
        
    def load_model(self):
        """加载TALLRec模型"""
        logger.info("Loading TALLRec recommendation model...")
        
        # 加载基础模型
        self.tokenizer = LlamaTokenizer.from_pretrained(self.config.model.base_model_path)
        self.base_model = LlamaForCausalLM.from_pretrained(
            self.config.model.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=getattr(self.config.model, 'load_in_8bit', False)
        )
        
        # 加载TALLRec适配器
        tallrec_adapter_path = getattr(self.config.model, 'tallrec_adapter_path', None)
        if tallrec_adapter_path and os.path.exists(tallrec_adapter_path):
            logger.info(f"Loading TALLRec adapter from: {tallrec_adapter_path}")
            self.model = PeftModel.from_pretrained(self.base_model, tallrec_adapter_path)
        else:
            logger.warning("TALLRec adapter not found, using base model")
            self.model = self.base_model
        
        # 设置模型配置
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.model.eval()
        self.tokenizer.padding_side = "left"
        
        logger.info("TALLRec recommendation model loaded successfully!")
    
    def construct_recommendation_prompt(self, user_history: List[str], target_item: str) -> str:
        """
        构造简化的推荐判断提示
        
        Args:
            user_history: 用户历史
            target_item: 目标物品
        
        Returns:
            recommendation_prompt: 推荐提示
        """
        if user_history:
            history_str = ", ".join(user_history[-5:])  # 取最近5个
            prompt = f"""### Instruction:
Based on the user's interaction history, would you recommend the target item? Answer with Yes or No.

### Input:
User History: {history_str}
Target Item: {target_item}

### Response:
"""
        else:
            prompt = f"""### Instruction:
Would you recommend the following item? Answer with Yes or No.

### Input:
Target Item: {target_item}

### Response:
"""
        
        return prompt
    
    def get_yes_no_probabilities(self, prompts: List[str]) -> List[Tuple[float, float]]:
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Getting Yes/No probabilities for {len(prompts)} prompts")
        
        all_probabilities = []
        batch_size = getattr(self.config.model, 'batch_size', 32)
        
        for batch_prompts in tqdm(
            batch_data(prompts, batch_size), 
            desc="Computing probabilities"
        ):
            # 批量分词
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                # 只需要一次前向传播，获取第一个生成token的概率分布
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=1,  # 只生成1个token
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=False,  # 确定性生成
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # 获取第一个生成token的概率分布
                first_token_scores = outputs.scores[0]  # [batch_size, vocab_size]
                
                # 提取Yes和No token的概率
                yes_no_logits = first_token_scores[:, [self.yes_token_id, self.no_token_id]]
                yes_no_probs = torch.softmax(yes_no_logits, dim=-1)  # [batch_size, 2]
                
                # 转换为列表
                batch_probabilities = []
                for i in range(yes_no_probs.shape[0]):
                    yes_prob = yes_no_probs[i, 0].item()
                    no_prob = yes_no_probs[i, 1].item()
                    batch_probabilities.append((yes_prob, no_prob))
                
                all_probabilities.extend(batch_probabilities)
        
        return all_probabilities
    
    def generate_recommendation_scores(self, data: List[Dict]) -> Tuple[List[Dict], Dict]:
        
        logger.info(f"Generating recommendation scores for {len(data)} samples")
        
        # 构造推荐提示
        prompts = []
        for sample in data:
            prompt = self.construct_recommendation_prompt(
                sample['user_history'],
                sample['item_title']
            )
            prompts.append(prompt)
        
        # 获取Yes/No概率
        probabilities = self.get_yes_no_probabilities(prompts)
        
        # 整理结果
        recommendation_scores = []
        for i, (sample, (yes_prob, no_prob)) in enumerate(zip(data, probabilities)):
            
            # 决策逻辑
            if yes_prob > no_prob:
                decision = "Yes"
                confidence_score = yes_prob
            else:
                decision = "No" 
                confidence_score = no_prob
            
            # decision_score用于AUC计算：Yes概率
            decision_score = yes_prob
            
            # 生成简化的推理
            reasoning = f"Based on model probability: Yes={yes_prob:.3f}, No={no_prob:.3f}"
            
            score_info = {
                'sample_index': i,
                'user_id': sample.get('user_id', ''),
                'item_id': sample.get('item_id', -1),
                'item_title': sample.get('item_title', ''),
                'true_label': sample.get('label', -1),
                'decision': decision,
                'reasoning': reasoning,
                'confidence_score': confidence_score,
                'decision_score': decision_score,  # 这个是AUC计算的关键！
                'yes_probability': yes_prob,
                'no_probability': no_prob
            }
            
            recommendation_scores.append(score_info)
        
        # 生成元数据以保持兼容性
        metadata = {
            'total_samples': len(recommendation_scores),
            'avg_yes_probability': float(np.mean([s['yes_probability'] for s in recommendation_scores])),
            'avg_no_probability': float(np.mean([s['no_probability'] for s in recommendation_scores])),
            'avg_decision_score': float(np.mean([s['decision_score'] for s in recommendation_scores])),
            'generation_method': 'simplified_probability_based'
        }
        
        return recommendation_scores, metadata
    
    def verify_token_ids(self):
        """
        验证Yes/No token ID是否正确
        """
        if self.tokenizer is None:
            logger.warning("Tokenizer not loaded, cannot verify token IDs")
            return
        
        # 检查token ID对应的文本
        yes_text = self.tokenizer.decode([self.yes_token_id])
        no_text = self.tokenizer.decode([self.no_token_id])
        
        logger.info(f"Token ID verification:")
        logger.info(f"  {self.yes_token_id} -> '{yes_text}'")
        logger.info(f"  {self.no_token_id} -> '{no_text}'")
        
        # 如果不是Yes/No，尝试自动查找
        if "yes" not in yes_text.lower() or "no" not in no_text.lower():
            logger.warning("Token IDs may be incorrect, attempting auto-detection...")
            
            # 尝试编码Yes和No来找到正确的token ID
            yes_tokens = self.tokenizer.encode("Yes", add_special_tokens=False)
            no_tokens = self.tokenizer.encode("No", add_special_tokens=False)
            
            if yes_tokens:
                self.yes_token_id = yes_tokens[0]
                logger.info(f"Auto-detected Yes token ID: {self.yes_token_id}")
            
            if no_tokens:
                self.no_token_id = no_tokens[0]
                logger.info(f"Auto-detected No token ID: {self.no_token_id}")
    
    def analyze_generation_quality(self, recommendation_scores: List[Dict]) -> Dict[str, float]:
        
        logger.info("Analyzing generation quality...")
        
        # 统计决策分布
        decisions = [score['decision'] for score in recommendation_scores]
        decision_counts = {
            'Yes': decisions.count('Yes'),
            'No': decisions.count('No')
        }
        
        # 计算平均概率
        yes_probs = [score['yes_probability'] for score in recommendation_scores]
        no_probs = [score['no_probability'] for score in recommendation_scores]
        
        avg_yes_prob = np.mean(yes_probs)
        avg_no_prob = np.mean(no_probs)
        
        # 计算决策一致性（如果有真实标签）
        true_labels = [score['true_label'] for score in recommendation_scores if score['true_label'] != -1]
        decision_scores = [score['decision_score'] for score in recommendation_scores if score['true_label'] != -1]
        
        consistency = 0.0
        if true_labels:
            predictions = [1 if score > 0.5 else 0 for score in decision_scores]
            consistency = sum(p == t for p, t in zip(predictions, true_labels)) / len(true_labels)
        
        # 计算概率分布的分离度
        prob_separation = np.mean([abs(y - n) for y, n in zip(yes_probs, no_probs)])
        
        quality_metrics = {
            'total_samples': len(recommendation_scores),
            'decision_distribution': decision_counts,
            'avg_yes_probability': float(avg_yes_prob),
            'avg_no_probability': float(avg_no_prob),
            'decision_consistency': float(consistency),
            'probability_separation': float(prob_separation),
            'confident_predictions_ratio': float(np.mean([max(y, n) > 0.7 for y, n in zip(yes_probs, no_probs)]))
        }
        
        return quality_metrics
    
    def run(self, data: List[Dict], output_path: str = None) -> Tuple[List[Dict], Dict]:
        """
        运行完整的推荐生成流程（简化版本）
        
        Args:
            data: 输入数据
            output_path: 输出路径（可选）
        
        Returns:
            recommendation_scores: 推荐评分
            quality_metrics: 质量指标
        """
        logger.info("=== Simplified TALLRec Recommendation Generation Started ===")
        
        try:
            # 1. 加载模型
            self.load_model()
            
            # 2. 验证token ID
            self.verify_token_ids()
            
            # 3. 生成推荐评分
            recommendation_scores, gen_metadata = self.generate_recommendation_scores(data)
            
            # 4. 分析生成质量
            quality_metrics = self.analyze_generation_quality(recommendation_scores)
            
            # 5. 保存结果
            if output_path:
                from utils import save_json
                save_json(recommendation_scores, output_path)
                metrics_path = output_path.replace('.json', '_metrics.json')
                save_json(quality_metrics, metrics_path)
                logger.info(f"Results saved to {output_path}")
            
            logger.info("=== Simplified TALLRec Recommendation Generation Completed ===")
            logger.info(f"Performance summary:")
            logger.info(f"  Total samples: {quality_metrics['total_samples']}")
            logger.info(f"  Avg Yes probability: {quality_metrics['avg_yes_probability']:.3f}")
            logger.info(f"  Decision consistency: {quality_metrics['decision_consistency']:.3f}")
            logger.info(f"  Confident predictions: {quality_metrics['confident_predictions_ratio']:.1%}")
            
            return recommendation_scores, quality_metrics
            
        except Exception as e:
            logger.error(f"Error during recommendation generation: {str(e)}")
            raise

def main():
    import argparse
    import os
    from utils import load_json
    
    parser = argparse.ArgumentParser(description="Simplified TALLRec Recommendation Generation")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--data_file", type=str, required=True, help="Input data file")
    parser.add_argument("--output_path", type=str, help="Output results path")
    
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
    
    try:
        # 加载数据
        data = load_json(args.data_file)
        logger.info(f"Loaded {len(data)} samples")
        
        # 初始化生成器
        generator = TALLRecRecommendationGenerator(config)
        
        # 运行生成
        recommendation_scores, quality_metrics = generator.run(data, args.output_path)
        
        # 计算AUC
        true_labels = [score['true_label'] for score in recommendation_scores if score['true_label'] != -1]
        decision_scores = [score['decision_score'] for score in recommendation_scores if score['true_label'] != -1]
        
        if true_labels and len(true_labels) > 0:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(true_labels, decision_scores)
            logger.info(f"�� AUC Score: {auc:.4f}")
        
        logger.info(f"✅ Success! Generated recommendation scores with simplified method")
        
    except Exception as e:
        logger.error(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()