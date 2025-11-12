#!/bin/bash
# 总并行运行脚本
# 用法：cd CnnModels/tests && ./run_all.sh
# 该脚本并行执行所有子目录（resnet, vgg等）的测试任务

set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Bi-Mask 并行测试集合启动"
echo "========================================"
echo "脚本位置: $SCRIPT_DIR"
echo ""

# 存储所有后台进程PID和对应的测试组名
declare -a PIDS=()
declare -a TEST_GROUPS=()

# 找到所有包含run.sh的子目录，并启动它们
for test_dir in "$SCRIPT_DIR"/*; do
    if [ -d "$test_dir" ] && [ -f "$test_dir/run.sh" ]; then
        test_name=$(basename "$test_dir")
        
        echo "[启动测试组] $test_name"
        
        # 在子目录中启动运行脚本，输出重定向到日志文件
        (
            cd "$test_dir"
            bash run.sh
        ) &> "$SCRIPT_DIR/${test_name}_group.log" &
        
        PIDS+=($!)
        TEST_GROUPS+=("$test_name")
        
        # 短暂延迟以避免启动过快
        sleep 2
    fi
done

echo ""
echo "========================================"
echo "所有测试组已在后台启动 (共 ${#PIDS[@]} 组)"
echo "========================================"
echo ""

# 监控所有测试组的执行进度
FAILED_GROUPS=()
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    GROUP=${TEST_GROUPS[$i]}
    
    echo "等待测试组完成: $GROUP (PID: $PID)"
    
    # 等待进程结束并获取返回值
    if wait $PID; then
        echo "✓ 测试组成功: $GROUP"
    else
        echo "✗ 测试组失败: $GROUP"
        FAILED_GROUPS+=("$GROUP")
    fi
done

echo ""
echo "========================================"
echo "所有测试组执行完成"
echo "========================================"
echo ""

# 输出最终测试结果摘要
if [ ${#FAILED_GROUPS[@]} -eq 0 ]; then
    echo "✓ 所有测试组执行成功！"
    echo ""
    echo "执行的测试组:"
    for group in "${TEST_GROUPS[@]}"; do
        echo "  ✓ $group"
    done
    exit 0
else
    echo "✗ 以下测试组执行失败:"
    for group in "${FAILED_GROUPS[@]}"; do
        echo "  ✗ $group (查看 ${group}_group.log 获取详情)"
    done
    echo ""
    echo "执行的测试组:"
    for group in "${TEST_GROUPS[@]}"; do
        if [[ " ${FAILED_GROUPS[@]} " =~ " ${group} " ]]; then
            echo "  ✗ $group"
        else
            echo "  ✓ $group"
        fi
    done
    exit 1
fi
