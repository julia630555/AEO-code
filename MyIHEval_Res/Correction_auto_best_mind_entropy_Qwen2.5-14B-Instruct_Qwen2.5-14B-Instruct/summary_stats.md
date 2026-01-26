# Evaluation Summary Statistics

Model: `best_mind_entropy_Qwen2.5-14B-Instruct_Qwen2.5-14B-Instruct`

| Dataset | Calibration Threshold | Final Detection Performance (Test Pool) | Detected | Corrected | CSR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **lang_detect_preprocessed_Qwen2.5-14B-Instruct_hd** | 0.5041 | PR-AUC: 0.00 \| FPR95: 0.00 | 0 | 0 | 0.00% |
| **rule-following_processed_Qwen2.5-14B-Instruct_hd** | 0.2987 | PR-AUC: 73.52 \| FPR95: 57.89 | 89 | 56 | 62.92% |
| **safety_processed_Qwen2.5-14B-Instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 0 | 0 | 0.00% |
| **translation_processed_Qwen2.5-14B-Instruct_hd** | 0.4750 | PR-AUC: 94.95 \| FPR95: 3.37 | 185 | 101 | 54.59% |
| **case_instruction_Qwen2.5-14B-Instruct_hd** | 0.0047 | PR-AUC: 96.03 \| FPR95: 74.19 | 126 | 126 | 100.00% |
| **digits_instruction_Qwen2.5-14B-Instruct_hd** | 0.0000 | PR-AUC: 0.00 \| FPR95: 0.00 | 104 | 104 | 100.00% |
| **json_instruction_Qwen2.5-14B-Instruct_hd** | 0.0000 | PR-AUC: 35.44 \| FPR95: 100.00 | 137 | 137 | 100.00% |
| **language_instruction_Qwen2.5-14B-Instruct_hd** | 0.0000 | PR-AUC: 0.00 \| FPR95: 0.00 | 122 | 122 | 100.00% |
| **list_instruction_Qwen2.5-14B-Instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 0 | 0 | 0.00% |
| **permutation_case_instruction_Qwen2.5-14B-Instruct_hd** | 0.0002 | PR-AUC: 47.90 \| FPR95: 88.41 | 123 | 123 | 100.00% |
| **permutation_digits_instruction_Qwen2.5-14B-Instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 0 | 0 | 0.00% |
| **permutation_json_instruction_Qwen2.5-14B-Instruct_hd** | 0.0000 | PR-AUC: 32.57 \| FPR95: 100.00 | 140 | 140 | 100.00% |
| **permutation_language_instruction_Qwen2.5-14B-Instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 0 | 0 | 0.00% |
| **permutation_list_instruction_Qwen2.5-14B-Instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 0 | 0 | 0.00% |
| **permutation_quotepresence_instruction_Qwen2.5-14B-Instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 0 | 0 | 0.00% |
| **permutation_sentencecount_instruction_Qwen2.5-14B-Instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 0 | 0 | 0.00% |
| **permutation_wordlength_instruction_Qwen2.5-14B-Instruct_hd** | 0.0000 | PR-AUC: 0.00 \| FPR95: 0.00 | 51 | 51 | 100.00% |
| **quotepresence_instruction_Qwen2.5-14B-Instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 1 | 1 | 100.00% |
| **sentencecount_instruction_Qwen2.5-14B-Instruct_hd** | 0.0145 | PR-AUC: 0.00 \| FPR95: 0.00 | 20 | 20 | 100.00% |
| **wordlength_instruction_Qwen2.5-14B-Instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 0 | 0 | 0.00% |

> [!NOTE]
> - **PR-AUC**: Precision-Recall Area Under Curve
> - **FPR95**: False Positive Rate at 95% True Positive Rate
> - **CSR**: Correction Success Rate (Corrected / Detected)