import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
from tqdm import tqdm

# 本地模型路径
model_dir = "model"  # 修改为您的模型路径

# 加载 tokenizer 和模型
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16) #.cuda()
    model = model.eval()
    tokenizer.pad_token_id = tokenizer.eos_token_id  # 显式设置 pad_token_id
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 加载数据
data = pd.read_json("20250208181531_camp_data_step_1_without_answer.jsonl", lines=True)
example_data = pd.read_json("20250214171329_提交示例.jsonl", lines=True)

# Prompt
system_prompt = """你是一位经验丰富的医疗专家，你需要根据给定的病例信息，判断病例的疾病 (diseases) 和原因 (reason)。

请按照以下格式返回结果：
diseases: 疾病名称
reason:  原因分析
"""

prompts = '''这是我的病例：
feature_content:
{}

请根据病例信息，给出疾病和原因。
'''

# 正则表达式提取函数
def re_get_diseases_bingli_json(text):
    diseases = re.findall(r'diseases:\s*(.*?)(?=\nreason:|$)', text, re.DOTALL)
    reasons = re.findall(r'reason:\s*(.*?)(?=$)', text, re.DOTALL)
    print(f"提取到的疾病列表: {diseases}")
    print(f"提取到的原因列表: {reasons}")
    if diseases and reasons:
        return diseases[0].strip(), reasons[0].strip()
    else:
        print(f"Warning: Could not extract diseases and reason from text: {text}")
        return "", ""

# 主循环
results = []

# 指定输出文件
output_file = "result2.jsonl"

# 确保文件存在，如果不存在则创建
with open(output_file, 'a', encoding='utf-8') as f:
    pass

for i, row in tqdm(enumerate(data.iterrows()), total=len(data)):
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompts.format(row[-1]['feature_content'])},
        ]
        tokenized_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_attention_mask=True  # 强制返回 attention_mask
        )
        print(f"tokenized_chat type: {type(tokenized_chat)}")
        if isinstance(tokenized_chat, torch.Tensor):
            tokenized_chat = {
                "input_ids": tokenized_chat,
                "attention_mask": torch.ones_like(tokenized_chat)
            }
        print(f"tokenized_chat keys: {tokenized_chat.keys()}")

        attention_mask = tokenized_chat.get('attention_mask')
        if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
            attention_mask = None

        generated_ids = model.generate(tokenized_chat["input_ids"], max_new_tokens=2048, attention_mask=attention_mask)

        #  正确处理 generated_ids
        generated_ids = generated_ids[:, tokenized_chat["input_ids"].shape[-1]:]


        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"模型输出: {response}") # 移动到这里
        diseases, reason = re_get_diseases_bingli_json(response)

        # 创建包含所有列的字典
        result = row[-1].to_dict()
        result['diseases'] = diseases
        result['reason'] = reason

        # 将结果写入 JSONL 文件
        with open(output_file, 'a', encoding='utf-8') as f:
            import json
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    except Exception as e:
        print(f"Error processing row: {row[-1]['feature_content']}")
        print(f"Error message: {e}")
        # 创建包含所有列的字典，并添加空字符串作为 diseases 和 reason
        result = row[-1].to_dict()
        result['diseases'] = ""
        result['reason'] = ""

        # 将结果写入 JSONL 文件
        with open(output_file, 'a', encoding='utf-8') as f:
            import json
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

print("处理完成，结果已保存到 result2.jsonl")