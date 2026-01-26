#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 MyIHEval_Res 中所有模型的统计信息
分为三类：
1. lang_detect, rule-following, safety, translation 这四个数据集的平均值
2. 其他数据集的平均值
3. 总体平均值
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_performance_metrics(perf_str: str) -> Tuple[float, float]:
    """
    解析 'PR-AUC: 99.92 | FPR95: 12.50' 格式的字符串
    返回 (PR-AUC, FPR95)
    """
    pr_auc_match = re.search(r'PR-AUC:\s*([\d.]+)', perf_str)
    fpr95_match = re.search(r'FPR95:\s*([\d.]+)', perf_str)
    
    pr_auc = float(pr_auc_match.group(1)) if pr_auc_match else 0.0
    fpr95 = float(fpr95_match.group(1)) if fpr95_match else 0.0
    
    return pr_auc, fpr95


def parse_csr(csr_str: str) -> float:
    """
    解析 '100.00%' 格式的字符串，返回百分比值
    """
    csr_match = re.search(r'([\d.]+)%', csr_str)
    return float(csr_match.group(1)) if csr_match else 0.0


def parse_summary_stats(file_path: str) -> Dict[str, Dict]:
    """
    解析 summary_stats.md 文件，提取每个数据集的指标
    """
    datasets = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找表格数据
    in_table = False
    for line in lines:
        line = line.strip()
        
        # 跳过表头和分隔符
        if line.startswith('| Dataset |') or line.startswith('| :---'):
            in_table = True
            continue
        
        if not in_table or not line.startswith('|'):
            continue
        
        # 先替换转义的 \| 为占位符，避免被当作分隔符
        line = line.replace('\\|', '<!PIPE!>')
        
        # 解析数据行
        parts = [p.strip() for p in line.split('|')]
        
        # 恢复占位符为 |
        parts = [p.replace('<!PIPE!>', '|') for p in parts]
        
        if len(parts) < 7:
            continue
        
        dataset_name_raw = parts[1].strip('*').strip()
        if not dataset_name_raw:
            continue
        
        # 提取基础数据集名称（去除后缀如 _preprocessed_xxx_hd 等）
        # 对于特殊数据集，如果名称以特殊数据集名开头，则使用基础名
        dataset_name = dataset_name_raw
        for base_name in ['lang_detect', 'rule-following', 'safety', 'translation']:
            if dataset_name_raw.startswith(base_name):
                dataset_name = base_name
                break
        
        perf_str = parts[3]
        csr_str = parts[6]
        
        pr_auc, fpr95 = parse_performance_metrics(perf_str)
        csr = parse_csr(csr_str)
        
        datasets[dataset_name] = {
            'PR-AUC': pr_auc,
            'FPR95': fpr95,
            'CSR': csr,
            'raw_name': dataset_name_raw  # 保存原始名称用于调试
        }
    
    return datasets


def calculate_averages(datasets: Dict[str, Dict], dataset_names: List[str]) -> Dict[str, float]:
    """
    计算指定数据集的平均值
    """
    if not dataset_names:
        return {'PR-AUC': 0.0, 'FPR95': 0.0, 'CSR': 0.0}
    
    total_pr_auc = 0.0
    total_fpr95 = 0.0
    total_csr = 0.0
    count = 0
    
    for name in dataset_names:
        if name in datasets:
            total_pr_auc += datasets[name]['PR-AUC']
            total_fpr95 += datasets[name]['FPR95']
            total_csr += datasets[name]['CSR']
            count += 1
    
    if count == 0:
        return {'PR-AUC': 0.0, 'FPR95': 0.0, 'CSR': 0.0}
    
    return {
        'PR-AUC': total_pr_auc / count,
        'FPR95': total_fpr95 / count,
        'CSR': total_csr / count
    }


def main():
    base_dir = Path(__file__).parent
    
    # 定义四个特殊数据集
    special_datasets = ['lang_detect', 'rule-following', 'safety', 'translation']
    
    # 查找所有 Correction_auto_* 文件夹
    model_dirs = [d for d in base_dir.iterdir() 
                  if d.is_dir() and d.name.startswith('Correction_auto_')]
    
    print(f"找到 {len(model_dirs)} 个模型文件夹:\n")
    
    all_results = []
    
    for model_dir in sorted(model_dirs):
        summary_file = model_dir / 'summary_stats.md'
        
        if not summary_file.exists():
            print(f"警告: {model_dir.name} 中没有找到 summary_stats.md")
            continue
        
        print(f"处理: {model_dir.name}")
        
        # 解析数据
        datasets = parse_summary_stats(str(summary_file))
        
        # 第一类：特殊四个数据集
        special_avg = calculate_averages(datasets, special_datasets)
        
        # 第二类：其他数据集
        other_datasets = [name for name in datasets.keys() 
                         if name not in special_datasets]
        other_avg = calculate_averages(datasets, other_datasets)
        
        # 第三类：所有数据集
        all_datasets = list(datasets.keys())
        total_avg = calculate_averages(datasets, all_datasets)
        
        result = {
            'model': model_dir.name,
            'special': special_avg,
            'other': other_avg,
            'total': total_avg,
            'special_count': len([n for n in special_datasets if n in datasets]),
            'other_count': len(other_datasets),
            'total_count': len(all_datasets)
        }
        
        all_results.append(result)
        
        # 输出结果
        print(f"\n  ① 特殊数据集 (lang_detect, rule-following, safety, translation) - {result['special_count']} 个:")
        print(f"     PR-AUC: {special_avg['PR-AUC']:.2f}")
        print(f"     FPR95:  {special_avg['FPR95']:.2f}")
        print(f"     CSR:    {special_avg['CSR']:.2f}%")
        
        print(f"\n  ② 其他数据集 - {result['other_count']} 个:")
        print(f"     PR-AUC: {other_avg['PR-AUC']:.2f}")
        print(f"     FPR95:  {other_avg['FPR95']:.2f}")
        print(f"     CSR:    {other_avg['CSR']:.2f}%")
        
        print(f"\n  ③ 总体平均值 - {result['total_count']} 个:")
        print(f"     PR-AUC: {total_avg['PR-AUC']:.2f}")
        print(f"     FPR95:  {total_avg['FPR95']:.2f}")
        print(f"     CSR:    {total_avg['CSR']:.2f}%")
        print("\n" + "="*80 + "\n")
    
    # 保存结果到文件
    output_file = base_dir / 'aggregated_statistics.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 聚合统计信息\n\n")
        
        for result in all_results:
            f.write(f"## {result['model']}\n\n")
            
            f.write("### ① 特殊数据集 (lang_detect, rule-following, safety, translation)\n\n")
            f.write(f"- 数据集数量: {result['special_count']}\n")
            f.write(f"- PR-AUC: {result['special']['PR-AUC']:.2f}\n")
            f.write(f"- FPR95: {result['special']['FPR95']:.2f}\n")
            f.write(f"- CSR: {result['special']['CSR']:.2f}%\n\n")
            
            f.write("### ② 其他数据集\n\n")
            f.write(f"- 数据集数量: {result['other_count']}\n")
            f.write(f"- PR-AUC: {result['other']['PR-AUC']:.2f}\n")
            f.write(f"- FPR95: {result['other']['FPR95']:.2f}\n")
            f.write(f"- CSR: {result['other']['CSR']:.2f}%\n\n")
            
            f.write("### ③ 总体平均值\n\n")
            f.write(f"- 数据集数量: {result['total_count']}\n")
            f.write(f"- PR-AUC: {result['total']['PR-AUC']:.2f}\n")
            f.write(f"- FPR95: {result['total']['FPR95']:.2f}\n")
            f.write(f"- CSR: {result['total']['CSR']:.2f}%\n\n")
            
            f.write("---\n\n")
        
        # 添加汇总表格
        f.write("## 汇总对比表\n\n")
        
        # 特殊数据集表格
        f.write("### 特殊数据集 (lang_detect, rule-following, safety, translation)\n\n")
        f.write("| 模型 | PR-AUC | FPR95 | CSR |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for result in all_results:
            model_name = result['model'].replace('Correction_auto_best_mind_entropy_', '')
            f.write(f"| {model_name} | {result['special']['PR-AUC']:.2f} | "
                   f"{result['special']['FPR95']:.2f} | {result['special']['CSR']:.2f}% |\n")
        f.write("\n")
        
        # 其他数据集表格
        f.write("### 其他数据集\n\n")
        f.write("| 模型 | PR-AUC | FPR95 | CSR |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for result in all_results:
            model_name = result['model'].replace('Correction_auto_best_mind_entropy_', '')
            f.write(f"| {model_name} | {result['other']['PR-AUC']:.2f} | "
                   f"{result['other']['FPR95']:.2f} | {result['other']['CSR']:.2f}% |\n")
        f.write("\n")
        
        # 总体表格
        f.write("### 总体平均值\n\n")
        f.write("| 模型 | PR-AUC | FPR95 | CSR |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for result in all_results:
            model_name = result['model'].replace('Correction_auto_best_mind_entropy_', '')
            f.write(f"| {model_name} | {result['total']['PR-AUC']:.2f} | "
                   f"{result['total']['FPR95']:.2f} | {result['total']['CSR']:.2f}% |\n")
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
