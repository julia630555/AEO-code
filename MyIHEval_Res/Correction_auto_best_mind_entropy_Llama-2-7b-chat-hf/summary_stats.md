# Evaluation Summary Statistics

Model: `best_mind_entropy_Llama-2-7b-chat-hf`

| Dataset | Calibration Threshold | Final Detection Performance (Test Pool) | Detected | Corrected | CSR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **case_instruction_label_hd** | 0.0288 | PR-AUC: 99.92 \| FPR95: 12.50 | 129 | 129 | 100.00% |
| **digits_instruction_label_hd** | 0.1065 | PR-AUC: 98.74 \| FPR95: 23.68 | 99 | 99 | 100.00% |
| **json_instruction_label_hd** | 0.1245 | PR-AUC: 100.00 \| FPR95: 0.00 | 133 | 133 | 100.00% |
| **language_instruction_label_hd** | 0.0000 | PR-AUC: 91.80 \| FPR95: 9.17 | 127 | 127 | 100.00% |
| **list_instruction_label_hd** | 0.5000 | PR-AUC: 0.00 \| FPR95: 0.00 | 0 | 0 | 0.00% |
| **permutation_case_instruction_label_hd** | 0.0000 | PR-AUC: 0.00 \| FPR95: 0.00 | 55 | 55 | 100.00% |
| **permutation_digits_instruction_label_hd** | 0.3787 | PR-AUC: 88.34 \| FPR95: 8.25 | 47 | 47 | 100.00% |
| **permutation_json_instruction_label_hd** | 0.0009 | PR-AUC: 99.90 \| FPR95: 1.45 | 84 | 84 | 100.00% |
| **permutation_language_instruction_label_hd** | 0.0000 | PR-AUC: 0.00 \| FPR95: 0.00 | 45 | 45 | 100.00% |
| **permutation_list_instruction_label_hd** | 0.0182 | PR-AUC: 98.69 \| FPR95: 0.00 | 83 | 83 | 100.00% |
| **permutation_quotepresence_instruction_label_hd** | 0.0000 | PR-AUC: 92.96 \| FPR95: 5.74 | 99 | 99 | 100.00% |
| **permutation_sentencecount_instruction_label_hd** | 0.4934 | PR-AUC: 0.00 \| FPR95: 0.00 | 2 | 2 | 100.00% |
| **permutation_wordlength_instruction_label_hd** | 0.4893 | PR-AUC: 98.43 \| FPR95: 16.67 | 74 | 74 | 100.00% |
| **quotepresence_instruction_label_hd** | 0.0009 | PR-AUC: 70.14 \| FPR95: 21.24 | 53 | 53 | 100.00% |
| **sentencecount_instruction_label_hd** | 0.3830 | PR-AUC: 0.00 \| FPR95: 0.00 | 130 | 130 | 100.00% |
| **wordlength_instruction_label_hd** | 0.3313 | PR-AUC: 0.00 \| FPR95: 0.00 | 6 | 6 | 100.00% |
| **lang_detect** | 0.4968 | PR-AUC: 100.00 \| FPR95: 0.00 | 741 | 131 | 17.68% |
| **rule-following** | 0.0000 | PR-AUC: 94.07 \| FPR95: 8.82 | 484 | 351 | 72.52% |
| **safety** | 0.0000 | PR-AUC: 97.29 \| FPR95: 16.15 | 668 | 668 | 100.00% |
| **translation** | 0.4837 | PR-AUC: 99.70 \| FPR95: 1.69 | 956 | 405 | 42.36% |

> [!NOTE]
> - **PR-AUC**: Precision-Recall Area Under Curve
> - **FPR95**: False Positive Rate at 95% True Positive Rate
> - **CSR**: Correction Success Rate (Corrected / Detected)