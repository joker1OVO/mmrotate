#!/bin/bash

# 用法: ./dota.sh <config> <exp_name> [val|test]
#   val  - 验证模式（评估 mAP），训练后默认执行
#   test - 测试模式（生成提交文件）
#
# 示例:
#   ./dota.sh oriented_rcnn_r50_fpn_1x_dota_le90_afe.py arfc_1 val
#   ./dota.sh oriented_rcnn_r50_fpn_1x_dota_le90_afe.py arfc_1 test

CONFIG="configs/afe/$1"
WORK_DIR="runs/DOTA/$2"
RESULT="$2"
MODE=${3:-val}   # 默认 val（训练完自动验证）

# 训练
python tools/train.py "$CONFIG" --work-dir "$WORK_DIR" --no-validate

if [ $? -ne 0 ]; then
    echo "Training failed. Evaluation skipped."
    exit 1
fi

if [ "$MODE" == "test" ]; then
    echo "=== Test mode: generating submission files ==="

    # 删除旧的 submission 目录（避免冲突）
    if [ -d "$WORK_DIR/$RESULT" ]; then
        echo "Removing old submission directory: $WORK_DIR/$RESULT"
        rm -rf "$WORK_DIR/$RESULT"
    fi

    python tools/test.py "$CONFIG" "$WORK_DIR/latest.pth" \
        --format-only \
        --eval-options "submission_dir=$WORK_DIR/$RESULT" \
        --cfg-options "data.test.test_mode=True"
else
    echo "=== Val mode: evaluating mAP ==="
    python tools/test.py "$CONFIG" "$WORK_DIR/latest.pth" --eval mAP
fi