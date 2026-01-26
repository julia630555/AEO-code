# 聚合统计信息

## Correction_auto_best_mind_entropy_Llama-2-7b-chat-hf

### ① 特殊数据集 (lang_detect, rule-following, safety, translation)

- 数据集数量: 4
- PR-AUC: 97.77
- FPR95: 6.67
- CSR: 58.14%

### ② 其他数据集

- 数据集数量: 16
- PR-AUC: 58.68
- FPR95: 6.17
- CSR: 93.75%

### ③ 总体平均值

- 数据集数量: 20
- PR-AUC: 66.50
- FPR95: 6.27
- CSR: 86.63%

---

## Correction_auto_best_mind_entropy_Mistral-7B-Instruct-v0.2_Mistral-7B-Instruct-v0.2

### ① 特殊数据集 (lang_detect, rule-following, safety, translation)

- 数据集数量: 4
- PR-AUC: 72.61
- FPR95: 1.32
- CSR: 21.85%

### ② 其他数据集

- 数据集数量: 16
- PR-AUC: 14.66
- FPR95: 24.09
- CSR: 100.00%

### ③ 总体平均值

- 数据集数量: 20
- PR-AUC: 26.25
- FPR95: 19.54
- CSR: 84.37%

---

## Correction_auto_best_mind_entropy_Phi3-mini-128k-instruct_Phi3-mini-128k-instruct

### ① 特殊数据集 (lang_detect, rule-following, safety, translation)

- 数据集数量: 4
- PR-AUC: 13.60
- FPR95: 21.15
- CSR: 22.28%

### ② 其他数据集

- 数据集数量: 16
- PR-AUC: 17.61
- FPR95: 18.03
- CSR: 100.00%

### ③ 总体平均值

- 数据集数量: 20
- PR-AUC: 16.80
- FPR95: 18.65
- CSR: 84.46%

---

## Correction_auto_best_mind_entropy_Qwen2.5-1.5B-Instruct_Qwen2.5-1.5B-Instruct

### ① 特殊数据集 (lang_detect, rule-following, safety, translation)

- 数据集数量: 4
- PR-AUC: 47.61
- FPR95: 9.21
- CSR: 22.38%

### ② 其他数据集

- 数据集数量: 16
- PR-AUC: 7.62
- FPR95: 12.50
- CSR: 100.00%

### ③ 总体平均值

- 数据集数量: 20
- PR-AUC: 15.61
- FPR95: 11.84
- CSR: 84.48%

---

## Correction_auto_best_mind_entropy_Qwen2.5-14B-Instruct_Qwen2.5-14B-Instruct

### ① 特殊数据集 (lang_detect, rule-following, safety, translation)

- 数据集数量: 4
- PR-AUC: 42.12
- FPR95: 15.31
- CSR: 29.38%

### ② 其他数据集

- 数据集数量: 16
- PR-AUC: 13.25
- FPR95: 22.66
- CSR: 56.25%

### ③ 总体平均值

- 数据集数量: 20
- PR-AUC: 19.02
- FPR95: 21.19
- CSR: 50.88%

---

## 汇总对比表

### 特殊数据集 (lang_detect, rule-following, safety, translation)

| 模型 | PR-AUC | FPR95 | CSR |
| :--- | :--- | :--- | :--- |
| Llama-2-7b-chat-hf | 97.77 | 6.67 | 58.14% |
| Mistral-7B-Instruct-v0.2_Mistral-7B-Instruct-v0.2 | 72.61 | 1.32 | 21.85% |
| Phi3-mini-128k-instruct_Phi3-mini-128k-instruct | 13.60 | 21.15 | 22.28% |
| Qwen2.5-1.5B-Instruct_Qwen2.5-1.5B-Instruct | 47.61 | 9.21 | 22.38% |
| Qwen2.5-14B-Instruct_Qwen2.5-14B-Instruct | 42.12 | 15.31 | 29.38% |

### 其他数据集

| 模型 | PR-AUC | FPR95 | CSR |
| :--- | :--- | :--- | :--- |
| Llama-2-7b-chat-hf | 58.68 | 6.17 | 93.75% |
| Mistral-7B-Instruct-v0.2_Mistral-7B-Instruct-v0.2 | 14.66 | 24.09 | 100.00% |
| Phi3-mini-128k-instruct_Phi3-mini-128k-instruct | 17.61 | 18.03 | 100.00% |
| Qwen2.5-1.5B-Instruct_Qwen2.5-1.5B-Instruct | 7.62 | 12.50 | 100.00% |
| Qwen2.5-14B-Instruct_Qwen2.5-14B-Instruct | 13.25 | 22.66 | 56.25% |

### 总体平均值

| 模型 | PR-AUC | FPR95 | CSR |
| :--- | :--- | :--- | :--- |
| Llama-2-7b-chat-hf | 66.50 | 6.27 | 86.63% |
| Mistral-7B-Instruct-v0.2_Mistral-7B-Instruct-v0.2 | 26.25 | 19.54 | 84.37% |
| Phi3-mini-128k-instruct_Phi3-mini-128k-instruct | 16.80 | 18.65 | 84.46% |
| Qwen2.5-1.5B-Instruct_Qwen2.5-1.5B-Instruct | 15.61 | 11.84 | 84.48% |
| Qwen2.5-14B-Instruct_Qwen2.5-14B-Instruct | 19.02 | 21.19 | 50.88% |
