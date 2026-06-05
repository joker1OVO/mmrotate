#!/bin/bash

# 用法: ./dota.sh <config> <exp_name> [val|test]
#   (无第三个参数)  - 训练 + 自动验证 mAP
#   val             - 仅验证 mAP（跳过训练，用已有 latest.pth）
#   test            - 仅测试（跳过训练，生成提交文件）
#
# 示例:
#   ./dota.sh oriented_rcnn_r50_fpn_1x_dota_le90_afe.py arfc_1        # 训练+验证
#   ./dota.sh oriented_rcnn_r50_fpn_1x_dota_le90_afe.py arfc_1 val    # 只验证
#   ./dota.sh oriented_rcnn_r50_fpn_1x_dota_le90_afe.py arfc_1 test   # 只测试

CONFIG="configs/afe/$1"
WORK_DIR="runs/DOTA/$2"
RESULT="$2"
MODE=${3:-train}   # 默认 train（训练 + 验证）

# 验证/测试优先用 runs 目录下的配置（训练时自动保存的副本）
TEST_CFG="$WORK_DIR/$1"
if [ ! -f "$TEST_CFG" ]; then
    echo "Warning: $TEST_CFG not found, fallback to $CONFIG"
    TEST_CFG="$CONFIG"
fi

if [ "$MODE" == "train" ]; then
    # ========== 训练 ==========
    python tools/train.py "$CONFIG" --work-dir "$WORK_DIR" --no-validate

    if [ $? -ne 0 ]; then
        echo "Training failed. Evaluation skipped."
        exit 1
    fi

    # 训练完后自动运行 val（用 val 集数据覆盖 data.test）
    echo "=== Training done, running val ==="
    python tools/test.py "$TEST_CFG" "$WORK_DIR/latest.pth" --eval mAP \
        --cfg-options data.test.ann_file=data/split_ss_dota/val/annfiles/ \
        data.test.img_prefix=data/split_ss_dota/val/images/

elif [ "$MODE" == "test" ]; then
    # ========== 仅测试（生成提交文件）==========
    echo "=== Test mode: generating submission files ==="

    if [ -d "$WORK_DIR/$RESULT" ]; then
        echo "Removing old submission directory: $WORK_DIR/$RESULT"
        rm -rf "$WORK_DIR/$RESULT"
    fi

    python tools/test.py "$TEST_CFG" "$WORK_DIR/latest.pth" \
        --format-only \
        --eval-options "submission_dir=$WORK_DIR/$RESULT" \
        --cfg-options "data.test.test_mode=True"
else
    # ========== 仅验证 mAP（用 val 集数据）==========
    echo "=== Val mode: evaluating mAP ==="
    python tools/test.py "$TEST_CFG" "$WORK_DIR/latest.pth" --eval mAP \
        --cfg-options data.test.ann_file=data/split_ss_dota/val/annfiles/ \
        data.test.img_prefix=data/split_ss_dota/val/images/
fi