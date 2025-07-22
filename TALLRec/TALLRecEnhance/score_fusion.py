

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support
from config import Config
from utils import setup_logging, normalize_scores

logger = setup_logging()

class ScoreFusion:
    def __init__(self, config: Config):
      
        self.config = config
        self.optimal_alpha = config.fusion.optimal_alpha
        self.fusion_method = config.fusion.fusion_method
      
    def normalize_scores(self, generation_scores: np.ndarray, 
                        distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        logger.info("Normalizing scores for fusion...")
      
        
        norm_generation = np.clip(generation_scores, 0, 1)
      
     
        valid_distances = distances[distances != float('inf')]
      
        if len(valid_distances) > 0:
            # 线性归一化：1 - (distance - min) / (max - min)
            max_dist = np.max(valid_distances)
            min_dist = np.min(valid_distances)
          
            norm_distances = np.zeros_like(distances)
            valid_mask = distances != float('inf')
          
            if max_dist > min_dist:
                norm_distances[valid_mask] = 1 - (distances[valid_mask] - min_dist) / (max_dist - min_dist + 1e-8)
            else:
                norm_distances[valid_mask] = 1.0
        else:
            norm_distances = np.zeros_like(distances)
      
        logger.info(f"Generation scores - Mean: {np.mean(norm_generation):.3f}, Std: {np.std(norm_generation):.3f}")
        logger.info(f"Distance scores - Mean: {np.mean(norm_distances):.3f}, Std: {np.std(norm_distances):.3f}")
      
        return norm_generation, norm_distances
  
    def linear_fusion(self, gen_scores: np.ndarray, dist_scores: np.ndarray, 
                     alpha: float = None) -> np.ndarray:
        
        if alpha is None:
            alpha = self.optimal_alpha
      
        fusion_scores = alpha * gen_scores + (1 - alpha) * dist_scores
        return fusion_scores
  
    def weighted_fusion(self, gen_scores: np.ndarray, dist_scores: np.ndarray,
                       alpha: float = None, confidence_weights: Optional[Tuple] = None) -> np.ndarray:
        
        if alpha is None:
            alpha = self.optimal_alpha
      
        if confidence_weights is None:
          
            gen_confidence = 2 * np.abs(gen_scores - 0.5)  # [0, 1]
            dist_confidence = 2 * np.abs(dist_scores - 0.5)
        else:
            gen_confidence, dist_confidence = confidence_weights
      
       
        total_confidence = gen_confidence + dist_confidence + 1e-8
        dynamic_alpha = alpha * gen_confidence / total_confidence
        dynamic_beta = (1 - alpha) * dist_confidence / total_confidence
      
       
        total_weight = dynamic_alpha + dynamic_beta + 1e-8
        dynamic_alpha /= total_weight
        dynamic_beta /= total_weight
      
        fusion_scores = dynamic_alpha * gen_scores + dynamic_beta * dist_scores
        return fusion_scores
  
    def adaptive_fusion(self, gen_scores: np.ndarray, dist_scores: np.ndarray,
                       agreement_threshold: float = None) -> np.ndarray:
        
        if agreement_threshold is None:
            agreement_threshold = self.config.fusion.confidence_threshold
      
    
        score_diff = np.abs(gen_scores - dist_scores)
        agreement = 1 - score_diff  # 差异越小，一致性越高
      
        base_alpha = 0.5
      
       
        gen_extremeness = 2 * np.abs(gen_scores - 0.5)
        dist_extremeness = 2 * np.abs(dist_scores - 0.5)
      
        # 自适应权重
        adaptive_weights = np.where(
            agreement > agreement_threshold,
            base_alpha,  # 一致时用平均权重
            gen_extremeness / (gen_extremeness + dist_extremeness + 1e-8)  # 不一致时偏向更确定的
        )
      
        fusion_scores = adaptive_weights * gen_scores + (1 - adaptive_weights) * dist_scores
      
        return fusion_scores
  
    def ensemble_fusion(self, gen_scores: np.ndarray, dist_scores: np.ndarray,
                       method: str = 'average') -> np.ndarray:
       
        
        linear_result = self.linear_fusion(gen_scores, dist_scores, alpha=0.6)
        weighted_result = self.weighted_fusion(gen_scores, dist_scores, alpha=0.6)
        adaptive_result = self.adaptive_fusion(gen_scores, dist_scores)
      
        if method == 'voting':
        
            linear_pred = (linear_result > 0.5).astype(int)
            weighted_pred = (weighted_result > 0.5).astype(int) 
            adaptive_pred = (adaptive_result > 0.5).astype(int)
          
          
            ensemble_pred = (linear_pred + weighted_pred + adaptive_pred) >= 2
            return ensemble_pred.astype(float)
          
        elif method == 'average':
         
            return (linear_result + weighted_result + adaptive_result) / 3
      
        elif method == 'max':
           
            return np.maximum(np.maximum(linear_result, weighted_result), adaptive_result)
      
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
  
    def calculate_confidence_weights(self, generation_scores: np.ndarray, 
                                   distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        gen_confidence = 2 * np.abs(generation_scores - 0.5)
      
        
        valid_distances = distances[distances != float('inf')]
        dist_confidence = np.zeros_like(distances)
      
        if len(valid_distances) > 0:
            mean_dist = np.mean(valid_distances)
            std_dist = np.std(valid_distances)
          
        
            valid_mask = distances != float('inf')
            if std_dist > 0:
                dist_confidence[valid_mask] = np.abs(distances[valid_mask] - mean_dist) / std_dist
                dist_confidence = np.clip(dist_confidence, 0, 1)
      
        return gen_confidence, dist_confidence
  
    def learn_optimal_weights(self, generation_scores: np.ndarray, 
                             distance_scores: np.ndarray, 
                             true_labels: List[int]) -> Tuple[float, float]:
        
        logger.info("Learning optimal fusion weights...")
      
        best_f1 = 0
        best_alpha = 0.5
        best_threshold = 0.5
      
       
        alpha_range = self.config.fusion.alpha_range
        threshold_range = self.config.fusion.threshold_range
      
        for alpha in alpha_range:
            for threshold in threshold_range:
             
                fusion_scores = self.linear_fusion(generation_scores, distance_scores, alpha)
              
            
                predictions = (fusion_scores > threshold).astype(int)
              
                # 计算F1分数
                _, _, f1, _ = precision_recall_fscore_support(
                    true_labels, predictions, average='binary', zero_division=0
                )
              
                if f1 > best_f1:
                    best_f1 = f1
                    best_alpha = alpha
                    best_threshold = threshold
      
        logger.info(f"Optimal weights: Generation={best_alpha:.2f}, Distance={1-best_alpha:.2f}")
        logger.info(f"Optimal threshold: {best_threshold:.2f}, F1: {best_f1:.4f}")
      
        
        self.optimal_alpha = best_alpha
        self.config.fusion.optimal_alpha = best_alpha
        self.config.fusion.decision_threshold = best_threshold
      
        return best_alpha, best_threshold
  
    def fuse_scores(self, generation_scores: np.ndarray, 
                   distances: np.ndarray) -> np.ndarray:
        
        logger.info(f"Fusing scores using method: {self.fusion_method}")
      
        
        norm_gen_scores, norm_dist_scores = self.normalize_scores(generation_scores, distances)
      
        
        valid_mask = ~np.isnan(norm_gen_scores) & ~np.isnan(norm_dist_scores) & (distances != float('inf'))
      
        if np.sum(valid_mask) == 0:
            logger.warning("No valid scores found!")
            return np.zeros_like(generation_scores)
      
     
        if np.sum(valid_mask) > 1:
            correlation = np.corrcoef(norm_gen_scores[valid_mask], norm_dist_scores[valid_mask])[0, 1]
            logger.info(f"Score correlation: {correlation:.3f}")
      
      
        if self.fusion_method == 'linear':
            final_scores = self.linear_fusion(norm_gen_scores, norm_dist_scores)
      
        elif self.fusion_method == 'weighted':
            
            gen_confidence, dist_confidence = self.calculate_confidence_weights(
                generation_scores, distances
            )
            final_scores = self.weighted_fusion(
                norm_gen_scores, norm_dist_scores, 
                confidence_weights=(gen_confidence, dist_confidence)
            )
      
        elif self.fusion_method == 'adaptive':
            final_scores = self.adaptive_fusion(norm_gen_scores, norm_dist_scores)
      
        elif self.fusion_method == 'ensemble':
            final_scores = self.ensemble_fusion(norm_gen_scores, norm_dist_scores, method='average')
      
        elif self.fusion_method == 'weighted_average':
           
            alpha = self.optimal_alpha  
            final_scores = self.linear_fusion(norm_gen_scores, norm_dist_scores, alpha)
            logger.info(f"✅ weighted_average融合: Gen={alpha:.3f}, Dist={1-alpha:.3f}")

        else:
            supported_methods = ['linear', 'weighted', 'adaptive', 'ensemble', 'weighted_average']
            raise ValueError(f"Unknown fusion method: {self.fusion_method}. Supported: {supported_methods}")
      
        
        final_scores = np.clip(final_scores, 0, 1)  # 确保在[0,1]范围
      
        logger.info(f"Final scores - Mean: {np.mean(final_scores):.3f}, Std: {np.std(final_scores):.3f}")
        logger.info(f"Score distribution: Min={np.min(final_scores):.3f}, Max={np.max(final_scores):.3f}")
      
        return final_scores
  
    def evaluate_fusion_quality(self, generation_scores: np.ndarray, 
                               distance_scores: np.ndarray,
                               fusion_scores: np.ndarray,
                               true_labels: List[int]) -> Dict[str, float]:
        
        from sklearn.metrics import roc_auc_score
      
        quality_metrics = {}
      
        # 计算各个组件的AUC
        try:
            quality_metrics['generation_auc'] = roc_auc_score(true_labels, generation_scores)
        except:
            quality_metrics['generation_auc'] = 0.0
      
        try:
            valid_mask = distance_scores != float('inf')
            if np.sum(valid_mask) > 0:
                quality_metrics['distance_auc'] = roc_auc_score(
                    np.array(true_labels)[valid_mask], 
                    distance_scores[valid_mask]
                )
            else:
                quality_metrics['distance_auc'] = 0.0
        except:
            quality_metrics['distance_auc'] = 0.0
      
        try:
            quality_metrics['fusion_auc'] = roc_auc_score(true_labels, fusion_scores)
        except:
            quality_metrics['fusion_auc'] = 0.0
      
       
        best_single_auc = max(quality_metrics['generation_auc'], quality_metrics['distance_auc'])
        quality_metrics['fusion_improvement'] = quality_metrics['fusion_auc'] - best_single_auc
      
  
        norm_gen, norm_dist = self.normalize_scores(generation_scores, np.where(distance_scores == float('inf'), 0, distance_scores))
        score_correlation = np.corrcoef(norm_gen, norm_dist)[0, 1] if len(norm_gen) > 1 else 0
        quality_metrics['score_correlation'] = score_correlation
      
        return quality_metrics

def main():
    import argparse
  
    parser = argparse.ArgumentParser(description="Multi-modal Score Fusion")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--generation_scores", type=str, required=True, help="Generation scores file")
    parser.add_argument("--distance_scores", type=str, required=True, help="Distance scores file")
    parser.add_argument("--labels_file", type=str, required=True, help="True labels file")
    parser.add_argument("--output_path", type=str, help="Output path for fusion results")
  
    args = parser.parse_args()
  
   
    if os.path.exists(args.config):
        config = Config.load(args.config)
    else:
        config = Config()
        logger.warning(f"Config file not found: {args.config}, using default config")
  
    try:
        
        generation_scores = np.load(args.generation_scores)
        distance_data = load_json(args.distance_scores)
        distances = np.array(distance_data['distances'])
        true_labels = load_json(args.labels_file)
      
        logger.info(f"Loaded {len(generation_scores)} generation scores")
        logger.info(f"Loaded {len(distances)} distance scores")
        logger.info(f"Loaded {len(true_labels)} labels")
      
        
        fusion = ScoreFusion(config)
      
        norm_gen, norm_dist = fusion.normalize_scores(generation_scores, distances)
        best_alpha, best_threshold = fusion.learn_optimal_weights(norm_gen, norm_dist, true_labels)
      
      
        fusion_scores = fusion.fuse_scores(generation_scores, distances)
      
     
        quality_metrics = fusion.evaluate_fusion_quality(
            generation_scores, norm_dist, fusion_scores, true_labels
        )
      
    
        if args.output_path:
            results = {
                'fusion_scores': fusion_scores.tolist(),
                'optimal_alpha': best_alpha,
                'optimal_threshold': best_threshold,
                'quality_metrics': quality_metrics,
                'fusion_method': config.fusion.fusion_method
            }
            save_json(results, args.output_path)
            logger.info(f"Fusion results saved to: {args.output_path}")
      
       
        print("\n=== Fusion Results ===")
        for metric, value in quality_metrics.items():
            print(f"{metric}: {value:.4f}")
      
    except Exception as e:
        logger.error(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()