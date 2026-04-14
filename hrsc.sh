#!/bin/bash

CONFIG="configs/afe/$1"
WORK_DIR="runs/hrsc/$2"
METRIC_VERSION=${3:-12}   # 默认使用 VOC2012

# 根据 metric_version 设置 use_07_metric 的值
if [ "$METRIC_VERSION" == "07" ]; then
    USE_07_METRIC="True"
    METRIC_NAME="VOC2007"
elif [ "$METRIC_VERSION" == "12" ]; then
    USE_07_METRIC="False"
    METRIC_NAME="VOC2012/COCO"
else
    echo "Error: metric_version must be '07' or '12' (default 12)"
    exit 1
fi

# 训练
python tools/train.py "$CONFIG" --work-dir "$WORK_DIR"

if [ $? -eq 0 ]; then
    # 测试：根据选择的指标进行评估
    echo "Evaluating with $METRIC_NAME metric..."
    python tools/test.py "$CONFIG" "$WORK_DIR/latest.pth" \
        --eval mAP \
        --eval-options "use_07_metric=$USE_07_METRIC" \
        --out "$WORK_DIR/results_${METRIC_VERSION}.pkl"
else
    echo "Training failed. Test skipped."
    exit 1
fi