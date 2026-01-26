# Evaluation Summary Statistics

Model: `best_mind_entropy_Mistral-7B-Instruct-v0.2_Mistral-7B-Instruct-v0.2`

| Dataset | Calibration Threshold | Final Detection Performance (Test Pool) | Detected | Corrected | CSR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **lang_detect_preprocessed_Mistral-7B-Instruct-v0.2_hd** | 0.4417 | PR-AUC: 99.70 \| FPR95: 0.00 | 216 | 71 | 32.87% |
| **rule-following_processed_Mistral-7B-Instruct-v0.2_hd** | 0.0070 | PR-AUC: 90.82 \| FPR95: 1.70 | 176 | 79 | 44.89% |
| **safety_processed_Mistral-7B-Instruct-v0.2_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 0 | 0 | 0.00% |
| **translation_processed_Mistral-7B-Instruct-v0.2_hd** | 0.5007 | PR-AUC: 99.91 \| FPR95: 3.57 | 1110 | 107 | 9.64% |
| **case_instruction_Mistral-7B-Instruct-v0.2_hd** | 0.0000 | PR-AUC: 0.00 \| FPR95: 0.00 | 139 | 139 | 100.00% |
| **digits_instruction_Mistral-7B-Instruct-v0.2_hd** | 0.0000 | PR-AUC: 0.00 \| FPR95: 0.00 | 139 | 139 | 100.00% |
| **json_instruction_Mistral-7B-Instruct-v0.2_hd** | 0.0000 | PR-AUC: 44.56 \| FPR95: 43.00 | 87 | 87 | 100.00% |
| **language_instruction_Mistral-7B-Instruct-v0.2_hd** | 0.0001 | PR-AUC: 58.04 \| FPR95: 74.26 | 131 | 131 | 100.00% |
| **list_instruction_Mistral-7B-Instruct-v0.2_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 14 | 14 | 100.00% |
| **permutation_case_instruction_Mistral-7B-Instruct-v0.2_hd** | 0.0326 | PR-AUC: 52.75 \| FPR95: 93.22 | 118 | 118 | 100.00% |
| **permutation_digits_instruction_Mistral-7B-Instruct-v0.2_hd** | 0.4562 | PR-AUC: 0.00 \| FPR95: 0.00 | 33 | 33 | 100.00% |
| **permutation_json_instruction_Mistral-7B-Instruct-v0.2_hd** | 0.0000 | PR-AUC: 49.25 \| FPR95: 75.00 | 122 | 122 | 100.00% |
| **permutation_language_instruction_Mistral-7B-Instruct-v0.2_hd** | 0.0001 | PR-AUC: 29.98 \| FPR95: 100.00 | 134 | 134 | 100.00% |
| **permutation_list_instruction_Mistral-7B-Instruct-v0.2_hd** | 0.0001 | PR-AUC: 0.00 \| FPR95: 0.00 | 110 | 110 | 100.00% |
| **permutation_quotepresence_instruction_Mistral-7B-Instruct-v0.2_hd** | 0.0013 | PR-AUC: 0.00 \| FPR95: 0.00 | 108 | 108 | 100.00% |
| **permutation_sentencecount_instruction_Mistral-7B-Instruct-v0.2_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 8 | 8 | 100.00% |
| **permutation_wordlength_instruction_Mistral-7B-Instruct-v0.2_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 21 | 21 | 100.00% |
| **quotepresence_instruction_Mistral-7B-Instruct-v0.2_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 22 | 22 | 100.00% |
| **sentencecount_instruction_Mistral-7B-Instruct-v0.2_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 17 | 17 | 100.00% |
| **wordlength_instruction_Mistral-7B-Instruct-v0.2_hd** | 0.0000 | PR-AUC: 0.00 \| FPR95: 0.00 | 133 | 133 | 100.00% |

> [!NOTE]
> - **PR-AUC**: Precision-Recall Area Under Curve
> - **FPR95**: False Positive Rate at 95% True Positive Rate
> - **CSR**: Correction Success Rate (Corrected / Detected)