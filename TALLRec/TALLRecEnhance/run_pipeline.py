"""
完整的TALLRec-Enhanced BIGRec运行流程 - 支持直接使用已处理数据
更新以支持增强版物品嵌入生成器
"""

import argparse
import os
import sys
from config import Config, default_config
from utils import setup_logging, set_seed, create_directory
from item_embedding_generator import EnhancedItemEmbeddingGenerator  # 更新导入
from enhanced_fusion_model import EnhancedFusionModel

def main():
    parser = argparse.ArgumentParser(description="TALLRec-Enhanced BIGRec Pipeline")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--mode", type=str, choices=["full", "embedding", "train", "evaluate"], 
                       default="full", help="Pipeline mode")
    parser.add_argument("--item_file", type=str, help="Item names file path (supports movie.dat format)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    
    # 支持指定已处理的数据文件
    parser.add_argument("--train_file", type=str, help="Processed train data file")
    parser.add_argument("--test_file", type=str, help="Processed test data file") 
    parser.add_argument("--valid_file", type=str, help="Processed valid data file")
    parser.add_argument("--data_dir", type=str, help="Directory containing train.json, test.json, valid.json")
    
    # 新增：支持增强版嵌入生成器的特殊参数
    parser.add_argument("--include_genres", action="store_true", 
                       help="Include genre information in movie titles (for movie.dat format)")
    parser.add_argument("--batch_size", type=int, help="Batch size for embedding generation")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info("=== TALLRec-Enhanced BIGRec Pipeline Started ===")
    
    # 加载配置
    if os.path.exists(args.config):
        config = Config.load(args.config)
        logger.info(f"Loaded config from: {args.config}")
    else:
        config = default_config
        logger.warning(f"Config file not found, using default config")
    
    # 命令行参数覆盖配置
    if args.include_genres:
        config.model.include_genres = True
    if args.batch_size:
        config.model.batch_size = args.batch_size
    
    # 设置随机种子
    set_seed(config.training.seed)
    
    # 创建输出目录
    create_directory(args.output_dir)
    
    try:
        if args.mode in ["full", "embedding"]:
            logger.info("Step 1: Generating enhanced item embeddings...")
            
            # 使用增强版物品嵌入生成器
            item_generator = EnhancedItemEmbeddingGenerator(config)
            
            # 确定物品文件路径
            item_file = args.item_file or config.data.item_names_file
            if not os.path.exists(item_file):
                logger.error(f"Item file not found: {item_file}")
                return
            
            # 检测并显示文件格式
            file_format = item_generator._detect_file_format(item_file)
            logger.info(f"Detected item file format: {file_format}")
            
            if file_format == "movie_dat":
                logger.info("�� Processing MovieLens movie.dat format")
                if hasattr(config.model, 'include_genres') and config.model.include_genres:
                    logger.info("�� Including genre information in embeddings")
            elif file_format == "standard":
                logger.info("�� Processing standard tab-separated format")
            
            # 运行增强版嵌入生成
            embeddings = item_generator.run(item_file)
            logger.info(f"Generated enhanced embeddings: {embeddings.shape}")
            
            # 输出嵌入文件信息
            embedding_path = config.model.item_embedding_path
            metadata_path = embedding_path.replace('.pt', '_metadata.json')
            logger.info(f"Embeddings saved to: {embedding_path}")
            logger.info(f"Metadata saved to: {metadata_path}")
        
        if args.mode in ["full", "train", "evaluate"]:
            logger.info("Step 2: Running enhanced fusion model...")
            
            # 确定数据文件路径
            if args.data_dir:
                # 使用指定的数据目录
                train_file = os.path.join(args.data_dir, "train.json")
                test_file = os.path.join(args.data_dir, "test.json")
                valid_file = os.path.join(args.data_dir, "valid.json")
                logger.info(f"Using data directory: {args.data_dir}")
            elif args.train_file and args.test_file:
                # 使用指定的单个文件
                train_file = args.train_file
                test_file = args.test_file
                valid_file = args.valid_file
                logger.info(f"Using specified data files: {train_file}, {test_file}")
            else:
                # 使用配置文件中的路径
                train_file = config.data.train_data_path
                test_file = config.data.test_data_path
                valid_file = None
                logger.info(f"Using config data paths: {train_file}, {test_file}")
            
            # 检查文件是否存在
            if not os.path.exists(train_file):
                logger.error(f"Train file not found: {train_file}")
                return
            if not os.path.exists(test_file):
                logger.error(f"Test file not found: {test_file}")
                return
            
            from utils import load_json
            train_data = load_json(train_file)
            test_data = load_json(test_file)
            
            logger.info(f"Loaded data: Train={len(train_data)}, Test={len(test_data)}")
            
            # 加载验证数据（如果存在）
            if valid_file and os.path.exists(valid_file):
                valid_data = load_json(valid_file)
                logger.info(f"Loaded validation data: {len(valid_data)}")
            else:
                valid_data = None
                logger.info("No validation data found, will use test data for validation")
            
            # 检查是否存在增强版嵌入文件
            embedding_path = config.model.item_embedding_path
            metadata_path = embedding_path.replace('.pt', '_metadata.json')
            
            if os.path.exists(embedding_path) and os.path.exists(metadata_path):
                logger.info("✅ Enhanced item embeddings found")
                # 加载并显示嵌入元数据
                from utils import load_json
                metadata = load_json(metadata_path)
                logger.info(f"�� Embedding info:")
                logger.info(f"   Items: {metadata['num_items']}")
                logger.info(f"   Embedding dim: {metadata['embedding_dim']}")
                logger.info(f"   File format: {metadata.get('file_format', 'unknown')}")
                logger.info(f"   Include genres: {metadata.get('include_genres', False)}")
            else:
                logger.warning("⚠️  Enhanced item embeddings not found, fusion model may need to generate them")
            
            # 运行增强融合模型
            model = EnhancedFusionModel(config)
            results_file = os.path.join(args.output_dir, "fusion_results.json")
            
            results = model.run_complete_pipeline(train_data, test_data, results_file)
            
            # 打印最终结果摘要
            print("\n" + "="*60)
            print("PIPELINE EXECUTION SUMMARY")
            print("="*60)
            
            eval_metrics = results['evaluation_metrics']
            print(f"�� Performance Metrics:")
            print(f"   Accuracy: {eval_metrics.get('accuracy', 0):.4f}")
            print(f"   Precision: {eval_metrics.get('precision', 0):.4f}")
            print(f"   Recall: {eval_metrics.get('recall', 0):.4f}")
            print(f"   F1-Score: {eval_metrics.get('f1', 0):.4f}")
            print(f"   AUC: {eval_metrics.get('fusion_auc', 0):.4f}")
            
            print(f"\n�� Model Configuration:")
            print(f"   Fusion Method: {config.fusion.fusion_method}")
            print(f"   Optimal Alpha: {config.fusion.optimal_alpha:.2f}")
            print(f"   Decision Threshold: {config.fusion.decision_threshold:.2f}")
            
            print(f"\n�� Data Files Used:")
            print(f"   Train: {train_file}")
            print(f"   Test: {test_file}")
            if valid_file and os.path.exists(valid_file):
                print(f"   Valid: {valid_file}")
            
            print(f"\n�� Output Files:")
            print(f"   Results: {results_file}")
            if args.mode == "full":
                print(f"   Item Embeddings: {config.model.item_embedding_path}")
                print(f"   Embedding Metadata: {metadata_path}")
            
            # 如果使用了增强版嵌入，显示额外信息
            if os.path.exists(metadata_path):
                metadata = load_json(metadata_path)
                print(f"\n�� Enhanced Embedding Info:")
                print(f"   File Format: {metadata.get('file_format', 'unknown')}")
                if metadata.get('file_format') == 'movie_dat':
                    print(f"   MovieLens Format: ✅")
                    print(f"   Include Genres: {metadata.get('include_genres', False)}")
                print(f"   ID Range: {min(metadata['index_to_id'].values())} - {max(metadata['index_to_id'].values())}")
        
        logger.info("=== Pipeline Completed Successfully! ===")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()