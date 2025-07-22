# create_config.py
from config import Config

# 创建默认配置
config = Config()

# 自定义配置
config.model.base_model_path = "/path/to/llama-7b"
config.model.tallrec_adapter_path = "/path/to/tallrec-adapter"
config.data.item_names_file = "./data/item_names.txt"
config.data.train_data_path = "./data/train.json"
config.data.test_data_path = "./data/test.json"

# 保存配置
config.save("config.json")
print("Default config saved to config.json")