import json
import re
from datasets import load_dataset

# 1. Load dataset và chọn 10000 ví dụ đầu
ds = load_dataset("meta-math/MetaMathQA", split='train')
sample = ds.select(range(10000))

# 2. Chuẩn bị regex để gãi đáp án
# Bắt phần sau "The answer is:"
pat = re.compile(r"The answer is:\s*([^\n]+)")

data = []
for idx, ex in enumerate(sample):
    full_solution = ex['response']
    
    # Tách đáp án ngắn
    m = pat.search(full_solution)
    short_answer = m.group(1).strip() if m else None

    data.append({
        "id": idx,
        "question": ex['query'],
        "solution": full_solution,
        "answer": short_answer
    })
    
# 3. Ghi ra file JSON
output_path = "sample_metamath.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Đã xuất {len(data)} ví dụ vào {output_path}")
