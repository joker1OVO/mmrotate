#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <config_filename> <work_dirname>"
    echo "  config_filename : 配置文件名称，如 oriented_rcnn_r50_fpn_1x_dota_le90.py"
    echo "  work_dirname    : 工作目录名称，如 oriented_rcnn（将自动放在 runs/DOTA/ 下）"
    echo ""
    echo "Example:"
    echo "  $0 oriented_rcnn_r50_fpn_1x_dota_le90.py oriented_rcnn"
    exit 1
fi

# 自动添加前缀路径
CONFIG="configs/afe/$1"
WORK_DIR="runs/DOTA/$2"
RESULT="$2"
# 训练
python tools/train.py "$CONFIG" --work-dir "$WORK_DIR" --no-validate

if [ $? -eq 0 ]; then
    # 测试前删除旧的 test_results 目录（避免冲突）
    if [ -d "$WORK_DIR/$RESULT" ]; then
        echo "Removing old test_results directory: $WORK_DIR/$RESULT"
        rm -rf "$WORK_DIR/$RESULT"
    fi

    # 测试
    python tools/test.py "$CONFIG" "$WORK_DIR/latest.pth" \
        --format-only \
        --eval-options "submission_dir=$WORK_DIR/$RESULT" \
        --cfg-options "data.test.test_mode=True"
else
    echo "Training failed. Test skipped."
    exit 1
fi