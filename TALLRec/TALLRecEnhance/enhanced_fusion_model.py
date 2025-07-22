

import torch
import numpy as np
import os
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
from config import Config
from utils import setup_logging, load_json, save_json, calculate_metrics, print_metrics


from tallrec_enhanced_encoder import TALLRecEnhancedEncoder
from tallrec_recommendation_generator import TALLRecRecommendationGenerator
from bigrec_distance_calculator import BIGRecTALLRecFusion
from score_fusion import ScoreFusion

logger = setup_logging()

class SharedModelManager:
    
    _shared_model = None
    _is_loaded = False
    
    @classmethod
    def get_or_create_model(cls, config: Config):
        
        if not cls._is_loaded:
            logger.info("�� Loading shared TALLRec model (one-time initialization)...")
            cls._shared_model = BIGRecTALLRecFusion(config)
            cls._is_loaded = True
            logger.info("✅ Shared TALLRec model loaded successfully!")
        else:
            logger.info("♻️  Reusing existing shared TALLRec model")
        
        return cls._shared_model
    
    @classmethod
    def reset(cls):
       
        cls._shared_model = None
        cls._is_loaded = False

class EnhancedFusionModel:
    def __init__(self, config: Config):
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # 获取共享的TALLRec模型
        self.shared_tallrec_model = SharedModelManager.get_or_create_model(config)
        
         # 初始化各个组件（使用原始构造函数）
        self.encoder = TALLRecEnhancedEncoder(config)
        self.generator = TALLRecRecommendationGenerator(config)
        self.calculator = self.shared_tallrec_model  # 直接使用共享的calculator
        self.fusion = ScoreFusion(config)
        
        # 运行时替换模型实例，避免重复加载
        self._inject_shared_models()
        
        # 模型状态
        self.is_calibrated = False
        self.optimal_threshold = 0.5

    def _inject_shared_models(self):
        
        logger.info("�� Injecting shared models to avoid duplicate loading...")
        
        # 检查并替换encoder中的模型实例
        if hasattr(self.encoder, 'model') or hasattr(self.encoder, 'tallrec_model'):
            # 替换encoder的模型实例
            if hasattr(self.encoder, 'model'):
                self.encoder.model = self.shared_tallrec_model.model
                self.encoder.tokenizer = self.shared_tallrec_model.tokenizer
                logger.info("✅ Replaced encoder model with shared instance")
            if hasattr(self.encoder, 'tallrec_model'):
                self.encoder.tallrec_model = self.shared_tallrec_model
                logger.info("✅ Replaced encoder tallrec_model with shared instance")
        
       
        if hasattr(self.generator, 'model') or hasattr(self.generator, 'tallrec_model'):
        
            if hasattr(self.generator, 'model'):
                self.generator.model = self.shared_tallrec_model.model
                self.generator.tokenizer = self.shared_tallrec_model.tokenizer
                logger.info("✅ Replaced generator model with shared instance")
            if hasattr(self.generator, 'tallrec_model'):
                self.generator.tallrec_model = self.shared_tallrec_model
                logger.info("✅ Replaced generator tallrec_model with shared instance")
        
     
        logger.info("�� All components now use shared model instances")

    def load_all_models(self):
  
        logger.info("Loading all models...")
        
        # 加载TALLRec增强编码器（如果需要额外初始化）
        if hasattr(self.encoder, 'load_model'):
            self.encoder.load_model()
        
        # 加载TALLRec推荐生成器（如果需要额外初始化）
        if hasattr(self.generator, 'load_model'):
            self.generator.load_model()
        
        # 加载物品嵌入（通过共享模型）
        if hasattr(self.calculator, 'load_item_embeddings'):
            self.calculator.load_item_embeddings()
        
        logger.info("All models loaded successfully!")

    def process_multimodal_features(self, data: List[Dict]) -> Dict[str, np.ndarray]:
        
        logger.info("=== Multi-modal Feature Processing Started ===")
        
      
        logger.info("Extracting enhanced user embeddings...")
        user_embeddings, embed_metadata = self.encoder.encode_user_preferences(data)
        
     
        logger.info("Generating recommendation scores...")
        recommendation_scores, gen_metadata = self.generator.generate_recommendation_scores(data)
        generation_scores = np.array([score['decision_score'] for score in recommendation_scores])
        
       
        logger.info("Calculating BIGRec distances...")
        item_indices = [sample['item_id'] for sample in data]
        distances, similarity_scores, calc_metadata = self.calculator.run(user_embeddings, item_indices)
        
        features = {
            'user_embeddings': user_embeddings.numpy() if torch.is_tensor(user_embeddings) else user_embeddings,
            'generation_scores': generation_scores,
            'distances': distances,
            'similarity_scores': similarity_scores,
            'item_indices': item_indices,
            'recommendation_details': recommendation_scores,
            'metadata': {
                'embedding': embed_metadata,
                'generation': gen_metadata,
                'calculation': calc_metadata
            }
        }
        
        logger.info("=== Multi-modal Feature Processing Completed ===")
        return features

    def calibrate_model(self, train_data: List[Dict]) -> Dict[str, float]:
       
        logger.info("=== Model Calibration Started ===")
        
      
        train_features = self.process_multimodal_features(train_data)
        train_labels = [sample['label'] for sample in train_data]
        
        
        norm_gen, norm_dist = self.fusion.normalize_scores(
            train_features['generation_scores'], 
            train_features['distances']
        )
        
        optimal_alpha, optimal_threshold = self.fusion.learn_optimal_weights(
            norm_gen, norm_dist, train_labels
        )
        
    
        self.optimal_threshold = optimal_threshold
        self.is_calibrated = True
        
        
        fusion_scores = self.fusion.fuse_scores(
            train_features['generation_scores'], 
            train_features['distances']
        )
        
        calibration_metrics = calculate_metrics(
            train_labels, 
            (fusion_scores > optimal_threshold).astype(int),
            fusion_scores
        )
        
        calibration_results = {
            'optimal_alpha': optimal_alpha,
            'optimal_threshold': optimal_threshold,
            'calibration_metrics': calibration_metrics
        }
        
        logger.info("=== Model Calibration Completed ===")
        print_metrics(calibration_metrics, "Calibration Results")
        
        return calibration_results

    def predict(self, data: List[Dict]) -> Dict[str, np.ndarray]:
        
        if not self.is_calibrated:
            logger.warning("Model not calibrated. Please call calibrate_model() first.")
        
        logger.info("=== Prediction Started ===")
        
       
        features = self.process_multimodal_features(data)
        
   
        fusion_scores = self.fusion.fuse_scores(
            features['generation_scores'], 
            features['distances']
        )
        
        
        predictions = (fusion_scores > self.optimal_threshold).astype(int)
        
        prediction_results = {
            'predictions': predictions,
            'fusion_scores': fusion_scores,
            'generation_scores': features['generation_scores'],
            'similarity_scores': features['similarity_scores'],
            'distances': features['distances'],
            'recommendation_details': features['recommendation_details'],
            'metadata': features['metadata']
        }
        
        logger.info("=== Prediction Completed ===")
        logger.info(f"Prediction distribution: Yes={np.sum(predictions)}, No={len(predictions)-np.sum(predictions)}")
        
        return prediction_results

    def evaluate(self, test_data: List[Dict]) -> Dict[str, float]:
        
        logger.info("=== Model Evaluation Started ===")
        
      
        prediction_results = self.predict(test_data)
        
       
        true_labels = [sample['label'] for sample in test_data]
        
       
        basic_metrics = calculate_metrics(
            true_labels,
            prediction_results['predictions'],
            prediction_results['fusion_scores']
        )
        
        
        component_metrics = self.fusion.evaluate_fusion_quality(
            prediction_results['generation_scores'],
            prediction_results['similarity_scores'],
            prediction_results['fusion_scores'],
            true_labels
        )
        
      
        detailed_metrics = self.analyze_prediction_quality(
            prediction_results, true_labels, test_data
        )
        
      
        evaluation_metrics = {
            **basic_metrics,
            **component_metrics,
            **detailed_metrics
        }
        
        logger.info("=== Model Evaluation Completed ===")
        print_metrics(evaluation_metrics, "Evaluation Results")
        
        return evaluation_metrics

    def analyze_prediction_quality(self, prediction_results: Dict, 
                                  true_labels: List[int], 
                                  test_data: List[Dict]) -> Dict[str, float]:
        
        detailed_metrics = {}
        
     
        history_lengths = [len(sample['user_history']) for sample in test_data]
        
      
        short_history_mask = np.array(history_lengths) <= 3
        if np.sum(short_history_mask) > 0:
            short_metrics = calculate_metrics(
                np.array(true_labels)[short_history_mask],
                prediction_results['predictions'][short_history_mask],
                prediction_results['fusion_scores'][short_history_mask]
            )
            detailed_metrics['short_history_f1'] = short_metrics['f1']
        
       
        long_history_mask = np.array(history_lengths) > 3
        if np.sum(long_history_mask) > 0:
            long_metrics = calculate_metrics(
                np.array(true_labels)[long_history_mask],
                prediction_results['predictions'][long_history_mask],
                prediction_results['fusion_scores'][long_history_mask]
            )
            detailed_metrics['long_history_f1'] = long_metrics['f1']
        
        
        high_confidence_mask = prediction_results['fusion_scores'] > 0.8
        low_confidence_mask = prediction_results['fusion_scores'] < 0.2
        
        detailed_metrics['high_confidence_ratio'] = np.mean(high_confidence_mask)
        detailed_metrics['low_confidence_ratio'] = np.mean(low_confidence_mask)
        
        if np.sum(high_confidence_mask) > 0:
            high_conf_accuracy = np.mean(
                prediction_results['predictions'][high_confidence_mask] == 
                np.array(true_labels)[high_confidence_mask]
            )
            detailed_metrics['high_confidence_accuracy'] = high_conf_accuracy
        
        
        gen_decisions = (prediction_results['generation_scores'] > 0.5).astype(int)
        dist_decisions = (prediction_results['similarity_scores'] > 0.5).astype(int)
        agreement_mask = gen_decisions == dist_decisions
        
        detailed_metrics['modality_agreement_ratio'] = np.mean(agreement_mask)
        
        if np.sum(agreement_mask) > 0:
            agreement_accuracy = np.mean(
                prediction_results['predictions'][agreement_mask] == 
                np.array(true_labels)[agreement_mask]
            )
            detailed_metrics['agreement_accuracy'] = agreement_accuracy
        
        return detailed_metrics

    def generate_recommendation_explanations(self, data: List[Dict], 
                                          prediction_results: Dict) -> List[Dict]:
        
        explanations = []
        
        for i, sample in enumerate(data):
            if 'user_id' not in sample:
                sample['user_id'] = f'user_{i:04d}'
            explanation = {
                'user_id': sample['user_id'],
                'item_id': sample['item_id'],
                'item_title': sample['item_title'],
                'prediction': int(prediction_results['predictions'][i]),
                'confidence': float(prediction_results['fusion_scores'][i]),
                'generation_score': float(prediction_results['generation_scores'][i]),
                'similarity_score': float(prediction_results['similarity_scores'][i]),
                'reasoning': prediction_results['recommendation_details'][i]['reasoning'],
                'user_history': sample['user_history'][-3:]  
            }
            
           
            if explanation['prediction'] == 1:
                explanation['recommendation'] = "Yes"
                explanation['explanation'] = f"Recommended with {explanation['confidence']:.2f} confidence based on user preferences and item similarity."
            else:
                explanation['recommendation'] = "No"
                explanation['explanation'] = f"Not recommended with {1-explanation['confidence']:.2f} confidence due to low user-item compatibility."
            
            explanations.append(explanation)
        
        return explanations

    def save_results(self, prediction_results: Dict, evaluation_metrics: Dict,
                    explanations: List[Dict], output_path: str):
       
     
        logger.info(f"Saving results to: {output_path}")
        
        # 准备保存数据
        results_data = {
            'model_config': self.config.to_dict(),
            'evaluation_metrics': evaluation_metrics,
            'prediction_summary': {
                'total_samples': len(prediction_results['predictions']),
                'positive_predictions': int(np.sum(prediction_results['predictions'])),
                'negative_predictions': int(len(prediction_results['predictions']) - np.sum(prediction_results['predictions'])),
                'average_confidence': float(np.mean(prediction_results['fusion_scores'])),
                'optimal_threshold': self.optimal_threshold
            },
            'detailed_predictions': {
                'predictions': prediction_results['predictions'].tolist(),
                'fusion_scores': prediction_results['fusion_scores'].tolist(),
                'generation_scores': prediction_results['generation_scores'].tolist(),
                'similarity_scores': prediction_results['similarity_scores'].tolist()
            },
            'explanations': explanations[:100],  # 保存前100个解释作为示例
            'fusion_config': {
                'method': self.config.fusion.fusion_method,
                'optimal_alpha': self.config.fusion.optimal_alpha,
                'decision_threshold': self.config.fusion.decision_threshold
            },
            'performance_metadata': prediction_results.get('metadata', {})
        }
        
        # 保存结果
        save_json(results_data, output_path)
        logger.info(f"Complete results saved successfully!")

    def run_complete_pipeline(self, train_data: List[Dict], 
                             test_data: List[Dict],
                             output_path: str = None) -> Dict[str, Any]:
        
        logger.info("=== Enhanced Fusion Model Pipeline Started ===")
        
        try:
           
            self.load_all_models()
            
         
            calibration_results = self.calibrate_model(train_data)
            
         
            evaluation_metrics = self.evaluate(test_data)
            
          
            prediction_results = self.predict(test_data)
            
          
            explanations = self.generate_recommendation_explanations(test_data, prediction_results)
            
        
            if output_path:
                self.save_results(prediction_results, evaluation_metrics, explanations, output_path)
            
     
            complete_results = {
                'calibration_results': calibration_results,
                'evaluation_metrics': evaluation_metrics,
                'prediction_results': prediction_results,
                'explanations': explanations
            }
            
            logger.info("=== Enhanced Fusion Model Pipeline Completed ===")
            return complete_results
            
        except Exception as e:
            logger.error(f"Error in complete pipeline: {str(e)}")
            raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Modular Enhanced TALLRec-BIGRec Fusion Model")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--train_data", type=str, required=True, help="Training data path")
    parser.add_argument("--test_data", type=str, required=True, help="Test data path")
    parser.add_argument("--output_path", type=str, help="Output path for results")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        config = Config.load(args.config)
    else:
        config = Config()
        logger.warning(f"Config file not found: {args.config}, using default config")
    
    # 命令行参数覆盖配置
    if args.batch_size:
        config.model.batch_size = args.batch_size
        logger.info(f"Using batch size: {args.batch_size}")
    
    # 检查输入文件
    for path in [args.train_data, args.test_data]:
        if not os.path.exists(path):
            logger.error(f"Data file not found: {path}")
            return
    
    try:
        # 加载数据
        train_data = load_json(args.train_data)
        test_data = load_json(args.test_data)
        
        logger.info(f"Loaded {len(train_data)} training samples")
        logger.info(f"Loaded {len(test_data)} test samples")
        
        # 初始化并运行模型
        model = EnhancedFusionModel(config)
        results = model.run_complete_pipeline(train_data, test_data, args.output_path)
        
        # 打印最终结果
        print("\n" + "="*50)
        print("FINAL RESULTS SUMMARY")
        print("="*50)
        
        eval_metrics = results['evaluation_metrics']
        print(f"Overall Performance:")
        print(f"  Accuracy: {eval_metrics.get('accuracy', 0):.4f}")
        print(f"  F1-Score: {eval_metrics.get('f1', 0):.4f}")
        print(f"  AUC: {eval_metrics.get('fusion_auc', 0):.4f}")
        
        print(f"\nComponent Performance:")
        print(f"  Generation AUC: {eval_metrics.get('generation_auc', 0):.4f}")
        print(f"  Distance AUC: {eval_metrics.get('distance_auc', 0):.4f}")
        print(f"  Fusion Improvement: {eval_metrics.get('fusion_improvement', 0):.4f}")
        
        print(f"\nModel Configuration:")
        print(f"  Fusion Method: {config.fusion.fusion_method}")
        print(f"  Optimal Alpha: {config.fusion.optimal_alpha:.2f}")
        print(f"  Decision Threshold: {config.fusion.decision_threshold:.2f}")
        
        logger.info("✅ Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()