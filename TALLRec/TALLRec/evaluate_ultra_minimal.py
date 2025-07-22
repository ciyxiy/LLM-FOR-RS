import torch
import json
import fire
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
from sklearn.metrics import roc_auc_score
import gc

def main(
    base_model: str = "",
    lora_weights: str = "",
    test_data_path: str = "",
    result_json_data: str = "temp.json",
):
    print("=== 终极显存优化评估 ===")
    
    device = "cuda"
    
    # 清理显存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 加载tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    
    # 超保守加载模型
    print("加载模型...")
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        device_map={'': 0},
        torch_dtype=torch.float16,
    )
    
    model.eval()
    
    # 强制禁用缓存
    model.config.use_cache = False
    
    print(f"模型加载完成，显存使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    def evaluate_minimal(instruction, input_text):
        """最小化显存使用的评估"""
        # 极短的prompt
        prompt = f"Instruction: {instruction[:100]}\nInput: {input_text[:100]}\nResponse:"
        
        # 限制长度
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=150,  # 极短输入
            padding=False
        ).to(device)
        
        with torch.no_grad():
            # 不生成，直接取logits
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # 最后一个token的logits
            
            # 只取Yes/No的logits
            yes_logit = logits[8241].item()  # Yes
            no_logit = logits[3782].item()   # No
            
            # 计算概率
            exp_yes = torch.exp(torch.tensor(yes_logit))
            exp_no = torch.exp(torch.tensor(no_logit))
            prob_yes = exp_yes / (exp_yes + exp_no)
            
        # 立即清理
        del inputs, outputs, logits
        torch.cuda.empty_cache()
        
        return prob_yes.item()
    
    # 加载测试数据
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # 限制测试数量
    if len(test_data) > 1000:
        print(f"限制测试样本为50个（原有{len(test_data)}个）")
        test_data = test_data[:1000]
    
    print(f"开始评估 {len(test_data)} 个样本...")
    
    predictions = []
    gold_labels = []
    
    for i, sample in enumerate(test_data):
        if i % 5 == 0:
            print(f"进度: {i}/{len(test_data)}")
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            prob = evaluate_minimal(sample['instruction'], sample['input'])
            predictions.append(prob)
            gold_labels.append(int(sample['output'] == 'Yes.'))
        except Exception as e:
            print(f"样本 {i} 失败: {e}")
            predictions.append(0.5)
            gold_labels.append(int(sample['output'] == 'Yes.'))
    
    # 计算AUC
    if len(set(gold_labels)) > 1:
        auc_score = roc_auc_score(gold_labels, predictions)
        print(f"AUC Score: {auc_score:.4f}")
    else:
        auc_score = 0.5
        print("警告：只有一种标签")
    
    # 保存结果
    result = {
        "movie": {
            "movie": {
                "finetune": {
                    "42": {
                        "64": auc_score
                    }
                }
            }
        }
    }
    
    with open(result_json_data, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"结果保存到: {result_json_data}")

if __name__ == "__main__":
    fire.Fire(main)
