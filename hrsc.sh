#!/bin/bash

CONFIG="configs/afe/$1"
WORK_DIR="runs/hrsc/$2"
MODE=${3:-07}   # 默认模式为 07

# 设置评估参数
if [ "$MODE" == "07" ]; then
    USE_07_METRIC="True"
    METRIC_NAME="VOC2007"
    DO_TRAIN=true
elif [ "$MODE" == "12" ]; then
    USE_07_METRIC="False"
    METRIC_NAME="VOC2012/COCO"
    DO_TRAIN=false
else
    echo "Error: mode must be '07' or '12' (default 07)"
    exit 1
fi

# 训练（仅在 mode=07 时执行）
if [ "$DO_TRAIN" = true ]; then
    python tools/train.py "$CONFIG" --work-dir "$WORK_DIR"
    if [ $? -ne 0 ]; then
        echo "Training failed. Test skipped."
        exit 1
    fi
fi

# 测试（两种模式都会执行）
# 检查模型文件是否存在
if [ ! -f "$WORK_DIR/latest.pth" ]; then
    echo "Error: Model file $WORK_DIR/latest.pth not found!"
    exit 1
fi

echo "Evaluating with $METRIC_NAME metric..."
python tools/test.py "$CONFIG" "$WORK_DIR/latest.pth" \
    --eval mAP \
    --eval-options "use_07_metric=$USE_07_METRIC" \
    --out "$WORK_DIR/results_${MODE}.pkl"