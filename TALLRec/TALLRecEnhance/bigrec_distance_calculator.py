import torch
import json
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import argparse
from config import Config
from sklearn.preprocessing import MinMaxScaler
class BIGRecTALLRecFusion:
    def __init__(self, config: Config):
      
        # 处理不同的调用方式
        if config is not None:
            # Pipeline调用方式：BIGRecTALLRecFusion(config)
            llama_model_path = config.model.base_model_path
            item_embedding_path = config.model.item_embedding_path
        elif llama_model_path is None or item_embedding_path is None:
            # 检查是否提供了必要参数
            raise ValueError("Either provide config object or both llama_model_path and item_embedding_path")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载LLaMA模型
        self.tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            llama_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        self.model.eval()
        self.tokenizer.padding_side = "left"
        
        # 加载预计算的物品嵌入 
        self.item_embeddings = torch.load(item_embedding_path).to(self.device)
        print(f"Loaded item embeddings: {self.item_embeddings.shape}")
        
        # 距离阈值（用于Yes/No判断）
        self.distance_threshold = None

        self.distance_scaler = MinMaxScaler()
        self.scaler_fitted = False
    
    def prepare_tallrec_data(self, tallrec_dataset_path):
        
        with open(tallrec_dataset_path, 'r') as f:
            data = json.load(f)
        
        processed_data = []
        for item in data:
            # 生成用户偏好描述文本（结合用户历史）
            user_pref_text = self.generate_user_preference_text(
                item['user_history'], 
                item['item_title']
            )
            
            processed_item = {
                'user_id': item['user_id'],
                'item_id': item['item_id'],
                'item_title': item['item_title'],
                'user_preference_text': user_pref_text,
                'label': item['label']
            }
            processed_data.append(processed_item)
        
        return processed_data
    
    def generate_user_preference_text(self, user_history, target_item):
        """
        基于用户历史生成偏好描述文本
        
        Args:
            user_history: 用户历史交互物品列表
            target_item: 目标物品
            
        Returns:
            生成的用户偏好描述文本
        """
        if len(user_history) > 5:
            recent_history = user_history[-5:]  # 取最近5个
        else:
            recent_history = user_history
        
        history_str = ", ".join(recent_history)
        
        
        prompt = f"""Based on the user's interaction history: {history_str}
The user is considering: {target_item}
Generate a preference description for this user in one sentence:"""
       
        preference_text = f"A user who enjoys {', '.join(recent_history[:3])} is considering {target_item}"
        
        return preference_text
    
    def get_user_embedding(self, user_texts, batch_size=32):
        """
        获取用户偏好文本的嵌入向量 (对应BIGRec Step 2中的oracle)
        
        Args:
            user_texts: 用户偏好描述文本列表
            batch_size: 批处理大小
            
        Returns:
            用户嵌入向量 [num_users, embedding_dim]
        """
        def batch_data(data, batch_size):
            for i in range(0, len(data), batch_size):
                yield data[i:i + batch_size]
        
        user_embeddings = []
        
        for batch_texts in tqdm(batch_data(user_texts, batch_size), desc="Computing user embeddings"):
            # 分词和编码
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            # 前向传播获取隐藏状态
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                # 取最后一层的最后一个token作为嵌入 (BIGRec方法)
                batch_embeddings = hidden_states[-1][:, -1, :].cpu()
                user_embeddings.append(batch_embeddings)
        
        return torch.cat(user_embeddings, dim=0)
    
    def calculate_l2_distances(self, user_embeddings, item_indices):
        """
        计算用户嵌入与对应物品嵌入的L2距离 (BIGRec Step 2核心)
        
        Args:
            user_embeddings: 用户嵌入 [num_samples, embedding_dim]
            item_indices: 对应的物品索引列表
            
        Returns:
            L2距离数组 [num_samples]
        """
        user_embeddings = user_embeddings.to(self.device)
        distances = []
        
        for i, item_idx in enumerate(item_indices):
            if item_idx < len(self.item_embeddings):
                # 计算 D_i = ||emb_i - oracle||_2
                item_emb = self.item_embeddings[item_idx]  # 物品嵌入
                user_emb = user_embeddings[i]  # 用户偏好嵌入 (oracle)
                user_emb = user_emb.to(item_emb.device)
                l2_dist = torch.norm(item_emb - user_emb, p=2).item()
                distances.append(l2_dist)
            else:
                distances.append(float('inf'))  # 无效物品索引
        
        return np.array(distances)
    
    def calibrate_threshold(self, distances, labels, method='optimal_f1'):
        distances_normalized = self.distance_scaler.fit_transform(distances.reshape(-1, 1)).flatten()
        self.scaler_fitted = True
       
        if method == 'optimal_f1':
            # 寻找F1分数最高的阈值
            thresholds = np.percentile(distances, np.arange(5, 100, 5))
            best_f1 = 0
            best_threshold = thresholds[0]
            
            for threshold in thresholds:
               
                predictions = (distances_normalized <= threshold).astype(int)
                _, _, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            self.distance_threshold = best_threshold
            return best_threshold
        
        elif method == 'median':
            self.distance_threshold = np.median(distances)
            return self.distance_threshold
    
    def predict_yes_no(self, user_texts, item_indices):
        """
        预测用户对物品的偏好 (Yes/No)
        
        Args:
            user_texts: 用户偏好描述文本列表
            item_indices: 对应的物品索引列表
            
        Returns:
            predictions: 预测结果 (1表示Yes, 0表示No)
            distances: L2距离数组
        """
        # 1. 获取用户嵌入
        user_embeddings = self.get_user_embedding(user_texts)
        
        # 2. 计算L2距离
        distances = self.calculate_l2_distances(user_embeddings, item_indices)

        
        if not self.scaler_fitted:
            raise ValueError("Distance scaler not fitted. Please call calibrate_threshold first.")
        
       
        distances_normalized = self.distance_scaler.transform(distances.reshape(-1, 1)).flatten()
        
        # 3. 基于阈值判断Yes/No
        if self.distance_threshold is None:
            raise ValueError("Distance threshold not set. Please call calibrate_threshold first.")
        
        predictions = (distances_normalized <= self.distance_threshold).astype(int)
        
        return predictions, distances
    
    def evaluate(self, test_data):
        """
        评估模型性能
        
        Args:
            test_data: 测试数据
            
        Returns:
            评估指标字典
        """
        user_texts = [item['user_preference_text'] for item in test_data]
        item_indices = [item['item_id'] for item in test_data]  # 假设item_id就是索引
        true_labels = [item['label'] for item in test_data]
        
        predictions, distances = self.predict_yes_no(user_texts, item_indices)
        
        # 计算评估指标
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
        
        # 使用负距离计算AUC
        auc = roc_auc_score(true_labels, -distances)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'threshold': self.distance_threshold
        }
        
        return metrics, predictions, distances
    def load_item_embeddings(self):
        """Pipeline期望的接口方法（实际上在__init__中已经加载了）"""
        pass  # 嵌入已经在__init__中加载
    
    def run(self, user_embeddings, item_indices):
        """Pipeline期望的接口方法"""
        distances = self.calculate_l2_distances(user_embeddings, item_indices)
        max_distance = np.max(distances) if len(distances) > 0 else 1.0
        similarity_scores = np.exp(-distances / max_distance)
        
        metadata = {
            'distance_method': 'l2',
            'num_samples': len(distances),
            'avg_distance': float(np.mean(distances)) if len(distances) > 0 else 0.0,
            'avg_similarity': float(np.mean(similarity_scores)) if len(similarity_scores) > 0 else 0.0
        }
        
        return distances, similarity_scores, metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama_model_path", type=str, required=True, help="LLaMA模型路径")
    parser.add_argument("--item_embedding_path", type=str, required=True, help="物品嵌入文件路径")
    parser.add_argument("--train_data_path", type=str, required=True, help="TALLRec训练数据路径")
    parser.add_argument("--test_data_path", type=str, required=True, help="TALLRec测试数据路径")
    parser.add_argument("--output_path", type=str, default="./fusion_results.json", help="结果输出路径")
    
    args = parser.parse_args()
    
    # 初始化融合模型
    fusion_model = BIGRecTALLRecFusion(
        llama_model_path=args.llama_model_path,
        item_embedding_path=args.item_embedding_path
    )
    
    # 准备数据
    print("Preparing training data...")
    train_data = fusion_model.prepare_tallrec_data(args.train_data_path)
    
    print("Preparing test data...")
    test_data = fusion_model.prepare_tallrec_data(args.test_data_path)
    
    # 在训练集上校准阈值
    print("Calibrating distance threshold...")
    train_user_texts = [item['user_preference_text'] for item in train_data]
    train_item_indices = [item['item_id'] for item in train_data]
    train_labels = [item['label'] for item in train_data]
    
    train_user_embeddings = fusion_model.get_user_embedding(train_user_texts)
    train_distances = fusion_model.calculate_l2_distances(train_user_embeddings, train_item_indices)
    
    optimal_threshold = fusion_model.calibrate_threshold(train_distances, train_labels, method='optimal_f1')
    print(f"Optimal threshold: {optimal_threshold}")
    
    # 在测试集上评估
    print("Evaluating on test set...")
    metrics, predictions, distances = fusion_model.evaluate(test_data)
    
    # 输出结果
    print("Evaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    # 保存详细结果
    results = {
        'metrics': metrics,
        'predictions': predictions.tolist(),
        'distances': distances.tolist(),
        'test_samples': len(test_data)
    }
    
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {args.output_path}")
# 在文件末尾添加以下代码


if __name__ == "__main__":
    main()