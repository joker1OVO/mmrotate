#!/bin/bash

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