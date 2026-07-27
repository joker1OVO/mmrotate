#!/bin/bash
export LD_PRELOAD="${CONDA_PREFIX}/lib/libcufft.so.10"

CONFIG="configs/safe/$1"
WORK_DIR="runs/hrsc/$2"
TEST_CONFIG="$WORK_DIR/$1"   # 测试用 runs 目录下的配置文件
MODE=${3:-7}   # 默认模式为 7 (VOC2007)

# 设置评估参数
if [ "$MODE" == "7" ] || [ "$MODE" == "07" ]; then
    USE_07_METRIC="True"
    METRIC_NAME="VOC2007"
    DO_TRAIN=true
elif [ "$MODE" == "12" ]; then
    USE_07_METRIC="False"
    METRIC_NAME="VOC2012/COCO"
    DO_TRAIN=false
else
    echo "Error: mode must be '7' (VOC2007) or '12' (VOC2012), default: 7"
    exit 1
fi

# 训练（仅在 VOC2007 模式时执行）
if [ "$DO_TRAIN" = true ]; then
    python tools/train.py "$CONFIG" --work-dir "$WORK_DIR"
    if [ $? -ne 0 ]; then
        echo "Training failed. Test skipped."
        exit 1
    fi
fi

# 测试（两种模式都会执行）
TEST_CFG="$TEST_CONFIG"
if [ ! -f "$TEST_CFG" ]; then
    echo "Warning: $TEST_CFG not found, fallback to $CONFIG"
    TEST_CFG="$CONFIG"
fi

if [ ! -f "$WORK_DIR/latest.pth" ]; then
    echo "Error: Model file $WORK_DIR/latest.pth not found!"
    exit 1
fi

echo "Evaluating with $METRIC_NAME metric..."
echo "Config: $TEST_CFG"
python tools/test.py "$TEST_CFG" "$WORK_DIR/latest.pth" \
    --eval mAP \
    --eval-options "use_07_metric=$USE_07_METRIC"