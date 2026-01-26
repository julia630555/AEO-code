# Evaluation Summary Statistics

Model: `best_mind_entropy_Qwen2.5-1.5B-Instruct_Qwen2.5-1.5B-Instruct`

| Dataset | Calibration Threshold | Final Detection Performance (Test Pool) | Detected | Corrected | CSR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **lang_detect_preprocessed_Qwen2.5-1.5B-Instruct_hd** | 0.4015 | PR-AUC: 0.00 \| FPR95: 0.00 | 1172 | 264 | 22.53% |
| **rule-following_processed_Qwen2.5-1.5B-Instruct_hd** | 0.0000 | PR-AUC: 90.50 \| FPR95: 35.02 | 221 | 99 | 44.80% |
| **safety_processed_Qwen2.5-1.5B-Instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 0 | 0 | 0.00% |
| **translation_processed_Qwen2.5-1.5B-Instruct_hd** | 0.4759 | PR-AUC: 99.95 \| FPR95: 1.80 | 1028 | 228 | 22.18% |
| **case_instruction_Qwen2.5-1.5B-Instruct_hd** | 0.0000 | PR-AUC: 0.00 \| FPR95: 0.00 | 139 | 139 | 100.00% |
| **digits_instruction_Qwen2.5-1.5B-Instruct_hd** | 0.4932 | PR-AUC: 0.00 \| FPR95: 0.00 | 5 | 5 | 100.00% |
| **json_instruction_Qwen2.5-1.5B-Instruct_hd** | 0.0000 | PR-AUC: 38.62 \| FPR95: 100.00 | 138 | 138 | 100.00% |
| **language_instruction_Qwen2.5-1.5B-Instruct_hd** | 0.2046 | PR-AUC: 0.00 \| FPR95: 0.00 | 53 | 53 | 100.00% |
| **list_instruction_Qwen2.5-1.5B-Instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 4 | 4 | 100.00% |
| **permutation_case_instruction_Qwen2.5-1.5B-Instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 8 | 8 | 100.00% |
| **permutation_digits_instruction_Qwen2.5-1.5B-Instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 5 | 5 | 100.00% |
| **permutation_json_instruction_Qwen2.5-1.5B-Instruct_hd** | 0.0000 | PR-AUC: 83.22 \| FPR95: 100.00 | 117 | 117 | 100.00% |
| **permutation_language_instruction_Qwen2.5-1.5B-Instruct_hd** | 0.0000 | PR-AUC: 0.00 \| FPR95: 0.00 | 133 | 133 | 100.00% |
| **permutation_list_instruction_Qwen2.5-1.5B-Instruct_hd** | 0.0000 | PR-AUC: 0.00 \| FPR95: 0.00 | 65 | 65 | 100.00% |
| **permutation_quotepresence_instruction_Qwen2.5-1.5B-Instruct_hd** | 0.0001 | PR-AUC: 0.00 \| FPR95: 0.00 | 52 | 52 | 100.00% |
| **permutation_sentencecount_instruction_Qwen2.5-1.5B-Instruct_hd** | 0.0000 | PR-AUC: 0.00 \| FPR95: 0.00 | 88 | 88 | 100.00% |
| **permutation_wordlength_instruction_Qwen2.5-1.5B-Instruct_hd** | 0.0000 | PR-AUC: 0.00 \| FPR95: 0.00 | 132 | 132 | 100.00% |
| **quotepresence_instruction_Qwen2.5-1.5B-Instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 3 | 3 | 100.00% |
| **sentencecount_instruction_Qwen2.5-1.5B-Instruct_hd** | 0.4919 | PR-AUC: 0.00 \| FPR95: 0.00 | 15 | 15 | 100.00% |
| **wordlength_instruction_Qwen2.5-1.5B-Instruct_hd** | 0.0000 | PR-AUC: 0.00 \| FPR95: 0.00 | 73 | 73 | 100.00% |

> [!NOTE]
> - **PR-AUC**: Precision-Recall Area Under Curve
> - **FPR95**: False Positive Rate at 95% True Positive Rate
> - **CSR**: Correction Success Rate (Corrected / Detected)