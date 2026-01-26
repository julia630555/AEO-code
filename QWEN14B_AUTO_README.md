# Qwen14b 自动评估系统

## 概述

此系统会自动监听 GPU 资源，当有足够的 GPU 空闲时自动运�?Qwen2.5-14B-Instruct 的评估任务�?
## 文件说明

- **`auto_run_qwen14b.py`** - 核心监听脚本，监�?GPU 并自动运行评�?- **`start_qwen14b_monitor.sh`** - 启动监听脚本的便捷命�?- **`check_qwen14b_status.sh`** - 检查评估状态和进度

## 配置参数

脚本会等待满足以下条件的 GPU�?
- **每个 GPU 需�?*: 至少 18 GB 空闲内存
- **GPU 数量**: 2 �?- **检查间�?*: �?60 秒检查一�?- **最大等待时�?*: 24 小时

## 使用方法

### 1. 启动监听

```bash

./start_qwen14b_monitor.sh


或者直接运行：

```bash

nohup python auto_run_qwen14b.py > gpu_monitor.log 2>&1 &


### 2. 检查状�?
```bash
./check_qwen14b_status.sh


### 3. 查看实时日志

**监听日志**（查�?GPU 检查情况）:
```bash



**评估日志**（当评估开始后�?
```bash



### 4. 停止监听

```bash
pkill -f auto_run_qwen14b.py


## 工作流程

1. **监听阶段**
   - 脚本�?60 秒检查一�?GPU 状�?   - 显示每个 GPU 的空闲内�?   - 等待找到 2 个满足条件的 GPU

2. **评估阶段**
   - 当找到满足条件的 GPU 后，自动启动 Qwen14b 评估
   - 使用选定�?GPU 运行 `MyCorrection.py`
   - 评估包括 4 个数据集�?     - lang_detect
     - rule-following
     - safety
     - translation

3. **统计阶段**
   - 评估完成后自动运行统计脚�?   - 更新聚合统计信息
   - 生成完整的评估报�?
## 输出结果

评估完成后，结果会保存在�?
- **评估结果**: 
- **汇总统�?*: 

## 当前状�?
查看当前 GPU 状态：
```bash
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv


从输出来看，目前没有单个 GPU 有超�?18GB 的空闲内存，所以监听脚本会持续等待直到 GPU 资源释放�?
## 结果处理

脚本默认输出目录�?`Measurement_auto`。为了让统计脚本正确识别 Qwen14b 的结果，评估完成后您需要手动重命名目录�?
```bash
# 1. 检查评估是否完�?./check_qwen14b_status.sh

# 2. 如果已完成（监听脚本退出），重命名目录
mv  \

# 3. 重新生成统计报告

python calculate_statistics.py


## 当前状�?
查看当前 GPU 状态：
