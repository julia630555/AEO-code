# Evaluation Summary Statistics

Model: `best_mind_entropy_Phi3-mini-128k-instruct_Phi3-mini-128k-instruct`

| Dataset | Calibration Threshold | Final Detection Performance (Test Pool) | Detected | Corrected | CSR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **lang_detect_preprocessed_Phi3-mini-128k-instruct_hd** | 0.4995 | PR-AUC: 0.00 \| FPR95: 0.00 | 1164 | 3 | 0.26% |
| **rule-following_processed_Phi3-mini-128k-instruct_hd** | 0.0001 | PR-AUC: 54.38 \| FPR95: 84.60 | 685 | 554 | 80.88% |
| **safety_processed_Phi3-mini-128k-instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 0 | 0 | 0.00% |
| **translation_processed_Phi3-mini-128k-instruct_hd** | 0.5032 | PR-AUC: 0.00 \| FPR95: 0.00 | 1179 | 94 | 7.97% |
| **case_instruction_Phi3-mini-128k-instruct_hd** | 0.0085 | PR-AUC: 97.52 \| FPR95: 100.00 | 119 | 119 | 100.00% |
| **digits_instruction_Phi3-mini-128k-instruct_hd** | 0.0001 | PR-AUC: 0.00 \| FPR95: 0.00 | 138 | 138 | 100.00% |
| **json_instruction_Phi3-mini-128k-instruct_hd** | 0.0139 | PR-AUC: 85.27 \| FPR95: 88.46 | 128 | 128 | 100.00% |
| **language_instruction_Phi3-mini-128k-instruct_hd** | 0.5056 | PR-AUC: 0.00 \| FPR95: 0.00 | 54 | 54 | 100.00% |
| **list_instruction_Phi3-mini-128k-instruct_hd** | 0.5088 | PR-AUC: 0.00 \| FPR95: 0.00 | 5 | 5 | 100.00% |
| **permutation_case_instruction_Phi3-mini-128k-instruct_hd** | 0.0003 | PR-AUC: 0.00 \| FPR95: 0.00 | 140 | 140 | 100.00% |
| **permutation_digits_instruction_Phi3-mini-128k-instruct_hd** | 0.0004 | PR-AUC: 0.00 \| FPR95: 0.00 | 133 | 133 | 100.00% |
| **permutation_json_instruction_Phi3-mini-128k-instruct_hd** | 0.0009 | PR-AUC: 98.89 \| FPR95: 100.00 | 133 | 133 | 100.00% |
| **permutation_language_instruction_Phi3-mini-128k-instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 118 | 118 | 100.00% |
| **permutation_list_instruction_Phi3-mini-128k-instruct_hd** | 0.0010 | PR-AUC: 0.00 \| FPR95: 0.00 | 129 | 129 | 100.00% |
| **permutation_quotepresence_instruction_Phi3-mini-128k-instruct_hd** | 0.0379 | PR-AUC: 0.00 \| FPR95: 0.00 | 85 | 85 | 100.00% |
| **permutation_sentencecount_instruction_Phi3-mini-128k-instruct_hd** | 0.0012 | PR-AUC: 0.00 \| FPR95: 0.00 | 129 | 129 | 100.00% |
| **permutation_wordlength_instruction_Phi3-mini-128k-instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 13 | 13 | 100.00% |
| **quotepresence_instruction_Phi3-mini-128k-instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 12 | 12 | 100.00% |
| **sentencecount_instruction_Phi3-mini-128k-instruct_hd** | 0.0020 | PR-AUC: 0.00 \| FPR95: 0.00 | 122 | 122 | 100.00% |
| **wordlength_instruction_Phi3-mini-128k-instruct_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 20 | 20 | 100.00% |

> [!NOTE]
> - **PR-AUC**: Precision-Recall Area Under Curve
> - **FPR95**: False Positive Rate at 95% True Positive Rate
> - **CSR**: Correction Success Rate (Corrected / Detected)