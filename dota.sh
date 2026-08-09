#!/bin/bash
export LD_PRELOAD="${CONDA_PREFIX}/lib/libcufft.so.10"

# 用法: ./dota.sh <config> <exp_name> [repeat_count|val|test]
#   (无第三个参数)  - 训练 + 测试 + 验证（1轮）
#   <数字>          - 重复训练 N 轮，版本号自动递增
#   val             - 仅验证 mAP（跳过训练）
#   test            - 仅测试（跳过训练，生成提交文件）
#
# 示例:
#   ./dota.sh oriented_rcnn_r50_fpn_1x_dota_le90_safe.py safe-1.1.1 3  # → safe-1.1.1/2/3
#   ./dota.sh oriented_rcnn_r50_fpn_1x_dota_le90_safe.py 1.1 3          # → 1.1, 1.2, 1.3
#   ./dota.sh oriented_rcnn_r50_fpn_1x_dota_le90_safe.py exp 3           # → exp1, exp2, exp3
#   ./dota.sh oriented_rcnn_r50_fpn_1x_dota_le90_safe.py arfc_1 val
#   ./dota.sh oriented_rcnn_r50_fpn_1x_dota_le90_safe.py arfc_1 test
#   ./dota.sh oriented_rcnn_r50_fpn_1x_dota_le90_safe.py arfc_1         # 训练1轮

CONFIG="configs/safe/$1"
CONFIG_FILE="$1"          # 训练时自动保存的配置文件名

# ---- 解析第三个参数: 数字=重复次数, val/test=模式, 空=默认训练 ----
if [[ "$3" =~ ^[0-9]+$ ]]; then
    REPEAT=$3
    MODE="train"
else
    REPEAT=1
    MODE=${3:-train}
fi

# ---- 解析版本号: "safe-1.1.1" → prefix="safe-1.1.", start=1 ----
BASE_NAME="$2"
PREFIX="${BASE_NAME%[0-9]*}"          # 去掉末尾数字
START_NUM="${BASE_NAME#$PREFIX}"      # 末尾数字

if [ -z "$START_NUM" ] || ! [[ "$START_NUM" =~ ^[0-9]+$ ]]; then
    NUM=1
    PREFIX="${BASE_NAME}"
    CURRENT_NAME="$BASE_NAME"
else
    NUM=$START_NUM
    CURRENT_NAME="$BASE_NAME"
fi

CURRENT="$CURRENT_NAME"

# ---- 训练/评估函数 ----
run_train() {
    local EXP="$1"
    local WORK_DIR="runs/DOTA/$EXP"
    local RESULT="$EXP"
    local TEST_CFG="$WORK_DIR/$CONFIG_FILE"

    # 训练
    python tools/train.py "$CONFIG" --work-dir "$WORK_DIR" --no-validate
    if [ $? -ne 0 ]; then
        echo "Training failed for $EXP."
        return 1
    fi

    # 训练时自动保存的配置
    if [ ! -f "$TEST_CFG" ]; then
        echo "Warning: $TEST_CFG not found, fallback to $CONFIG"
        TEST_CFG="$CONFIG"
    fi

    # 测试（生成提交文件）
    if [ -d "$WORK_DIR/$RESULT" ]; then
        rm -rf "$WORK_DIR/$RESULT"
    fi
    echo "=== Test: $EXP ==="
    python tools/test.py "$TEST_CFG" "$WORK_DIR/latest.pth" \
        --format-only \
        --eval-options "submission_dir=$WORK_DIR/$RESULT" \
        --cfg-options "data.test.test_mode=True" "data.test.filter_empty_gt=False"

    # 验证 mAP
    echo "=== Val: $EXP ==="
    python tools/test.py "$TEST_CFG" "$WORK_DIR/latest.pth" --eval mAP \
        --cfg-options data.test.ann_file=data/ss_dota/val/annfiles/ \
        data.test.img_prefix=data/ss_dota/val/images/
}

run_test() {
    local EXP="$1"
    local WORK_DIR="runs/DOTA/$EXP"
    local RESULT="$EXP"
    local TEST_CFG="$WORK_DIR/$CONFIG_FILE"

    if [ ! -f "$TEST_CFG" ]; then
        echo "Warning: $TEST_CFG not found, fallback to $CONFIG"
        TEST_CFG="$CONFIG"
    fi

    if [ -d "$WORK_DIR/$RESULT" ]; then
        rm -rf "$WORK_DIR/$RESULT"
    fi
    echo "=== Test: $EXP ==="
    python tools/test.py "$TEST_CFG" "$WORK_DIR/latest.pth" \
        --format-only \
        --eval-options "submission_dir=$WORK_DIR/$RESULT" \
        --cfg-options "data.test.test_mode=True" "data.test.filter_empty_gt=False"
}

run_val() {
    local EXP="$1"
    local WORK_DIR="runs/DOTA/$EXP"
    local TEST_CFG="$WORK_DIR/$CONFIG_FILE"

    if [ ! -f "$TEST_CFG" ]; then
        echo "Warning: $TEST_CFG not found, fallback to $CONFIG"
        TEST_CFG="$CONFIG"
    fi

    echo "=== Val: $EXP ==="
    python tools/test.py "$TEST_CFG" "$WORK_DIR/latest.pth" --eval mAP \
        --cfg-options data.test.ann_file=data/ss_dota/val/annfiles/ \
        data.test.img_prefix=data/ss_dota/val/images/
}

# ---- 执行 ----
for ((i=1; i<=REPEAT; i++)); do
    echo ""
    echo "============================================================"
    echo "  [$i/$REPEAT] $CURRENT"
    echo "============================================================"

    case "$MODE" in
        train)
            run_train "$CURRENT"
            if [ $? -ne 0 ]; then
                echo "Aborting remaining runs."
                exit 1
            fi
            ;;
        test)
            run_test "$CURRENT"
            ;;
        val)
            run_val "$CURRENT"
            ;;
        *)
            echo "Unknown mode: $MODE"
            exit 1
            ;;
    esac

    # 递增版本号
    NUM=$((NUM + 1))
    CURRENT="${PREFIX}${NUM}"
done
