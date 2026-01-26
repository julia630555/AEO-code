#!/bin/bash
# Qwen14b 自动评估启动脚本

echo "======================================================"
echo "Qwen14b 自动评估监听脚本"
echo "======================================================"
echo ""
echo "此脚本将监听 GPU 资源，当有足够的空闲 GPU 时自动运行评估。"
echo ""
echo "配置："
echo "  - 每个 GPU 需要: 18 GB 空闲内存"
echo "  - 需要 GPU 数量: 2 个"
echo "  - 检查间隔: 60 秒"
echo "  - 最大等待时间: 24 小时"
echo ""
echo "日志文件："
echo ""
echo "======================================================"
echo ""

# 检查是否已经有监听脚本在运行
if pgrep -f "auto_run_qwen14b.py" > /dev/null; then
    echo "⚠ 监听脚本已经在运行中"
    echo ""
    echo "进程信息:"
    ps aux | grep auto_run_qwen14b | grep -v grep
    echo ""
    echo "查看监听日志:"
    echo 
    echo ""
    echo "停止监听脚本:"
    echo "  pkill -f auto_run_qwen14b.py"
else
    echo "启动监听脚本..."
    nohup python auto_run_qwen14b.py > gpu_monitor.log 2>&1 &
    
    sleep 2
    
    if pgrep -f "auto_run_qwen14b.py" > /dev/null; then
        echo "✓ 监听脚本已启动"
        echo ""
        echo "进程信息:"
        ps aux | grep auto_run_qwen14b | grep -v grep
        echo ""
        echo "实时查看监听日志:"
        echo ""
        echo "停止监听:"
        echo "  pkill -f auto_run_qwen14b.py"
    else
        echo "✗ 启动失败，请检查日志"
    fi
fi

echo ""
echo "======================================================"
