"""
增强版物品嵌入生成器：使用LLaMA模型为物品生成向量表示
支持多种文件格式，包括movie.dat格式
"""

import torch
import os
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
import argparse
from typing import List, Tuple
from config import Config
from utils import setup_logging, batch_data, save_json, create_directory, get_file_size, load_json

logger = setup_logging()

class EnhancedItemEmbeddingGenerator:
    def __init__(self, config: Config):
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """加载LLaMA模型"""
        logger.info(f"Loading LLaMA model from: {self.config.model.base_model_path}")
        
        self.tokenizer = LlamaTokenizer.from_pretrained(self.config.model.base_model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            self.config.model.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=self.config.model.load_in_8bit
        )
        
        # 设置token配置
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.model.eval()
        self.tokenizer.padding_side = "left"
        
        logger.info("Model loaded successfully!")
    
    def _detect_file_format(self, file_path: str) -> str:
        
        if 'movies.dat' in file_path:
            return "movie_dat"
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取前几行来检测格式
            for _ in range(5):
                line = f.readline().strip()
                if line:
                    if '::' in line and len(line.split('::')) >= 2:
                       
                        first_part = line.split('::')[0].strip()
                        try:
                            int(first_part)
                            return "movie_dat"
                        except ValueError:
                            pass
                    elif '\t' in line and len(line.split('\t')) >= 2:
                        return "standard"
        
      
        return "standard"
    
    def load_item_names(self, item_file_path: str) -> Tuple[List[str], List[int]]:
       
        logger.info(f"Loading item names from: {item_file_path}")
        
        item_names = []
        item_ids = []
        
        # 检测文件格式
        file_format = self._detect_file_format(item_file_path)
        logger.info(f"Detected file format: {file_format}")
        
        with open(item_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    if file_format == "movie_dat":
                        # movie.dat格式: ID::Title::Genres
                        parts = line.split('::')
                        if len(parts) >= 2:
                            item_id = int(parts[0].strip())
                            item_name = parts[1].strip()
                            
                            # 可选：包含类型信息
                            if len(parts) >= 3 and hasattr(self.config.model, 'include_genres') and self.config.model.include_genres:
                                genres = parts[2].strip()
                                item_name = f"{item_name} [{genres}]"  # 将类型信息加入标题
                            
                            item_names.append(item_name)
                            item_ids.append(item_id)
                        else:
                            logger.warning(f"Invalid movie.dat format at line {line_num}: {line}")
                            
                    elif file_format == "standard":
                        # 标准格式: item_name\titem_id
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            item_name = parts[0].strip(' ').strip('"')
                            item_id = int(parts[1])
                            item_names.append(item_name)
                            item_ids.append(item_id)
                        else:
                            logger.warning(f"Invalid standard format at line {line_num}: {line}")
                            
                except ValueError as e:
                    logger.warning(f"Invalid item_id at line {line_num}: {e}")
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
        
        logger.info(f"Loaded {len(item_names)} valid items")
        
        # 显示示例
        logger.info("Sample items:")
        for i in range(min(5, len(item_names))):
            logger.info(f"  {item_ids[i]}: {item_names[i]}")
        
        return item_names, item_ids
    
    def generate_embeddings(self, item_names: List[str]) -> torch.Tensor:
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Generating embeddings for {len(item_names)} items")
        logger.info(f"Batch size: {self.config.model.batch_size}")
        
        item_embeddings = []
        
        for batch_items in tqdm(
            batch_data(item_names, self.config.model.batch_size), 
            desc="Generating embeddings"
        ):
            # 分词和编码
            inputs = self.tokenizer(
                batch_items, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.config.model.max_length
            )
            
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True
                )
                
                hidden_states = outputs.hidden_states
                # 提取最后一层最后一个token的嵌入
                batch_embeddings = hidden_states[-1][:, -1, :].detach().cpu()
                item_embeddings.append(batch_embeddings)
        
        # 拼接所有批次的嵌入
        item_embeddings = torch.cat(item_embeddings, dim=0)
        
        logger.info(f"Generated embeddings shape: {item_embeddings.shape}")
        logger.info(f"Embedding dtype: {item_embeddings.dtype}")
        
        return item_embeddings
    
    def save_embeddings(self, embeddings: torch.Tensor, item_ids: List[int], 
                       item_names: List[str]):
        """保存嵌入和元数据"""
        # 保存嵌入张量
        embedding_path = self.config.model.item_embedding_path
        create_directory(os.path.dirname(embedding_path))
        
        logger.info(f"Saving embeddings to: {embedding_path}")
        torch.save(embeddings, embedding_path)
        
        # 保存增强版元数据
        metadata_path = embedding_path.replace('.pt', '_metadata.json')
        metadata = {
            'num_items': len(item_names),
            'embedding_dim': embeddings.shape[1],
            'embedding_dtype': str(embeddings.dtype),
            'model_path': self.config.model.base_model_path,
            'max_length': self.config.model.max_length,
            'include_genres': getattr(self.config.model, 'include_genres', False),
            'file_format': getattr(self, '_detected_format', 'unknown'),
            'item_mapping': {
                item_id: {'name': name, 'index': idx} 
                for idx, (item_id, name) in enumerate(zip(item_ids, item_names))
            },
            # 新增：ID到索引的快速映射
            'id_to_index': {
                item_id: idx 
                for idx, item_id in enumerate(item_ids)
            },
            # 新增：索引到ID的快速映射
            'index_to_id': {
                idx: item_id 
                for idx, item_id in enumerate(item_ids)
            }
        }
        
        save_json(metadata, metadata_path)
        
        # 打印统计信息
        file_size = get_file_size(embedding_path)
        logger.info(f"Embeddings saved successfully!")
        logger.info(f"File size: {file_size}")
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"ID range: {min(item_ids)} - {max(item_ids)}")
        logger.info(f"Total unique IDs: {len(set(item_ids))}")
    
    def run(self, item_file_path: str = None) -> torch.Tensor:
        """
        运行完整的嵌入生成流程
        
        Args:
            item_file_path: 物品文件路径（可选，使用配置中的默认值）
        
        Returns:
            embeddings: 生成的嵌入张量
        """
        logger.info("=== Enhanced Item Embedding Generation Started ===")
        
        # 使用提供的路径或配置中的默认路径
        if item_file_path is None:
            item_file_path = self.config.data.item_names_file
        
        try:
            
            self.load_model()
            
          
            item_names, item_ids = self.load_item_names(item_file_path)
          
            self._detected_format = self._detect_file_format(item_file_path)
            
            if not item_names:
                raise ValueError("No valid items found in the file")
            
          
            embeddings = self.generate_embeddings(item_names)
            
        
            self.save_embeddings(embeddings, item_ids, item_names)
            
            logger.info("=== Enhanced Item Embedding Generation Completed ===")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error during embedding generation: {str(e)}")
            raise


class ItemEmbeddingGenerator(EnhancedItemEmbeddingGenerator):
    """原始ItemEmbeddingGenerator的增强版替代"""
    pass

def main():
    parser = argparse.ArgumentParser(description="Generate item embeddings using LLaMA model")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    parser.add_argument("--item_file", type=str, help="Path to item names file (supports movie.dat)")
    parser.add_argument("--output_path", type=str, help="Output path for embeddings")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing")
    parser.add_argument("--include_genres", action="store_true", 
                       help="Include genre information in movie titles (for movie.dat)")
    
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        config = Config.load(args.config)
    else:
        config = Config()
        logger.warning(f"Config file not found: {args.config}, using default config")
    
  
    if args.item_file:
        config.data.item_names_file = args.item_file
    if args.output_path:
        config.model.item_embedding_path = args.output_path
    if args.batch_size:
        config.model.batch_size = args.batch_size
    if args.include_genres:
        config.model.include_genres = True
    
  
    if not os.path.exists(config.data.item_names_file):
        logger.error(f"Item file not found: {config.data.item_names_file}")
        return
    
    if not os.path.exists(config.model.base_model_path):
        logger.error(f"Model path not found: {config.model.base_model_path}")
        return
    
  
    try:
        generator = EnhancedItemEmbeddingGenerator(config)
        embeddings = generator.run()
        logger.info(f"✅ Success! Generated embeddings for {embeddings.shape[0]} items")
        
        # 打印文件格式信息
        detected_format = getattr(generator, '_detected_format', 'unknown')
        logger.info(f"�� Detected file format: {detected_format}")
        
        if detected_format == "movie_dat":
            logger.info("�� Successfully processed MovieLens movie.dat format")
        elif detected_format == "standard":
            logger.info("�� Successfully processed standard tab-separated format")
        
    except Exception as e:
        logger.error(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

def demo_usage():
    """使用示例"""
    config = Config()
    
   
    print("=== Processing movie.dat file ===")
    generator = EnhancedItemEmbeddingGenerator(config)
    
   
    embeddings = generator.run("movie.dat")
    print(f"Generated embeddings: {embeddings.shape}")
    
  
    print("\n=== Processing standard format file ===")
    embeddings = generator.run("item_names.txt")
    print(f"Generated embeddings: {embeddings.shape}")

if __name__ == "__main__":
    main()