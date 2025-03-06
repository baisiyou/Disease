Rubing: # AI-Assisted Medical Diagnosis and Disease Prediction

## Overview
This project focuses on developing AI models for **medical diagnosis and disease prediction** using real-world clinical data, medical literature, and virtual patient records. The goal is to improve the accuracy and interpretability of disease diagnosis while ensuring broad coverage across six major healthcare data categories.

We utilize **Qwen2.5-7B-Instruct** for model training and evaluation, leveraging techniques such as:
- **Pre-training** on medical datasets
- **Supervised Fine-Tuning (SFT)** to enhance diagnostic accuracy
- **Prompt learning** for optimized inference under open-source AI constraints

This project aims to enhance model generalization, making AI-assisted medical diagnosis more reliable and accessible.

---

## 🏗️ Implementation Details
The project is implemented in **Python** using **Hugging Face Transformers** and **PyTorch** for model inference. The primary script, `main3.py`, performs the following tasks:
- Loads the **Qwen2.5-7B-Instruct** model from a local directory
- Processes clinical case data from JSONL files
- Uses **LLM-based prompt engineering** to extract disease and reason analysis
- Parses model output using **regular expressions**
- Stores predictions in a structured JSONL format

### **Files in this repository:**
- `main3.py`: Main script for medical diagnosis prediction.
- `20250208181531_camp_data_step_1_without_answer.jsonl`: Input dataset containing clinical case features.
- `20250214171329_提交示例.jsonl`: Sample expected output for reference.
- `result2.jsonl`: Output file with disease predictions and explanations.
- `model/`: Directory containing the locally stored **Qwen2.5-7B-Instruct** model.

---

## 🚀 Model Execution Workflow
### **1️⃣ Load Model and Data**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_dir = "model"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.eval()
tokenizer.pad_token_id = tokenizer.eos_token_id
```

### **2️⃣ Define Prompt and Generate Predictions**
```python
system_prompt = """
You are an experienced medical expert. Based on the given clinical case, determine the disease(s) and cause(s).

Format:
diseases: <disease_name>
reason: <reasoning>
"""

prompts = """
Here is a patient case:
feature_content:
{}

Based on the case details, provide the disease and reason.
"""
```

### **3️⃣ Model Inference and Output Parsing**
```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompts.format(row['feature_content'])},
]

tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_attention_mask=True
)

response = model.generate(tokenized_chat["input_ids"], max_new_tokens=2048)
output_text = tokenizer.decode(response[0], skip_special_tokens=True)
```

### **4️⃣ Extract Disease and Reason**
```python
import re

def extract_disease_reason(text):
    diseases = re.findall(r'diseases:\s*(.*?)(?=\nreason:|$)', text, re.DOTALL)
    reasons = re.findall(r'reason:\s*(.*?)(?=$)', text, re.DOTALL)
    return diseases[0].strip(), reasons[0].strip() if diseases and reasons else ("", "")

predicted_disease, predicted_reason = extract_disease_reason(output_text)
```

### **5️⃣ Save Predictions**
```python
import json
result = {"feature_content": row['feature_content'], "diseases": predicted_disease, "reason": predicted_reason}
with open("result2.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(result, ensure_ascii=False) + '\n')
```

---

## 🔄 Running the Model
### **1️⃣ Ensure Dependencies Are Installed**
```bash
pip install torch transformers pandas tqdm
```

### **2️⃣ Run the Script**
```bash
python main3.py
```

### **3️⃣ Review Output**
Results are saved in `result2.jsonl` in JSONL format:
```jsonl
{"feature_content": "Patient case details...", "diseases": "Hypertension", "reason": "Elevated blood pressure detected."}
```

---

## 📜 Future Enhancements
- Expand dataset coverage with **multi-language clinical data**.
- Integrate **Federated Learning** for privacy-preserving AI training.
- Improve interpretability with **Explainable AI (XAI) techniques**.

---

## 📩 Contact
For questions or contributions, please reach out via the project discussion forum.

Let’s build better AI for healthcare! 🚀

