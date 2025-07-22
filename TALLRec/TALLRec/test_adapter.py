import torch

try:
weights = torch.load('/media/jyh/09e3ad80-13b7-403c-907b-75fe5213b4d4/Ycx/alpaca-lora-7B/adapter_model.bin', map_location='cpu')
print('✓ 文件加载成功')
print(f'包含的keys: {list(weights.keys())[:5]}...')
print(f'文件大小信息: {len(weights)} 个参数组')
except Exception as e:
print(f'✗ 文件加载失败: {e}')
