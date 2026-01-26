#!/bin/bash
# Qwen2.5-14B-Instruct 完整训练和评估流程

set -e  # 遇到错误立即退出

echo "=================================================="
echo "Step 1: 训练 MLP 检测器 for Qwen2.5-14B-Instruct"
echo "=================================================="

python MYtrain_newdata.py \
  --data_paths \
    
  --output_path \
  --gpu 4 \
  --epochs 20 \
  --entropy_weight 1.0 \
  --model_name "best_mind_entropy.pt"

echo ""
echo "=================================================="
echo "Step 2: 运行评估和矫正 for Qwen2.5-14B-Instruct"
echo "=================================================="

# 检查生成的模型文件名
MODEL_FILE=$(ls -t )
if [ -z "$MODEL_FILE" ]; then
    echo "错误：未找到训练好的模型文件！"
    exit 1
fi

echo "使用模型: $MODEL_FILE"

python MyCorrection.py \
  --llm_path  \
  --detector_path "$MODEL_FILE" \
  --eval_files \
  --output_dir  \
  --gpu 2 4 \
  --cal_ratio 0.3

echo ""
echo "=================================================="
echo "Step 3: 更新聚合统计信息"
echo "=================================================="

python 

echo ""
echo "=================================================="
echo "全部完成！"
echo "=================================================="
