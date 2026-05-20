#!/bin/bash

set -e

echo "======================================"
echo " SegSky Demo: Build + Query"
echo "======================================"

# 进入项目根目录，也就是 SSegsky
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[1/5] Current project root:"
echo "$PROJECT_ROOT"

# 数据路径
BASE_DATA="$PROJECT_ROOT/sample_data/base_10k.fvecs"
QUERY_DATA="$PROJECT_ROOT/sample_data/query_1k.fvecs"

# 输出目录
RESULT_DIR="$PROJECT_ROOT/results"
mkdir -p "$RESULT_DIR"

echo "[2/5] Check sample data..."

if [ ! -f "$BASE_DATA" ]; then
    echo "Error: base data not found: $BASE_DATA"
    exit 1
fi

if [ ! -f "$QUERY_DATA" ]; then
    echo "Error: query data not found: $QUERY_DATA"
    exit 1
fi

echo "Base data:  $BASE_DATA"
echo "Query data: $QUERY_DATA"

echo "[3/5] Build project..."

mkdir -p build
cd build

cmake ..
cmake --build . -j

cd "$PROJECT_ROOT"

echo "[4/5] Find executable..."

# 根据你的项目结构自动寻找可执行文件
# 如果你的真实可执行文件名字不是这些，需要在这里改
if [ -f "$PROJECT_ROOT/build/query_seg_hnsw" ]; then
    EXEC="$PROJECT_ROOT/build/query_seg_hnsw"
elif [ -f "$PROJECT_ROOT/build/crosswin" ]; then
    EXEC="$PROJECT_ROOT/build/crosswin"
elif [ -f "$PROJECT_ROOT/crosswin" ]; then
    EXEC="$PROJECT_ROOT/crosswin"
else
    echo "Error: executable file not found."
    echo "Please check the generated binary name in build/."
    echo "You can run:"
    echo "  ls build"
    exit 1
fi

echo "Executable: $EXEC"

echo "[5/5] Run query demo..."

OUTPUT_FILE="$RESULT_DIR/demo_result.txt"

# 注意：
# 下面这一段参数需要根据你 crosswin / query 程序真实支持的参数改。
# 如果你的程序不是 --data --query 这种格式，就把这一段换成你的真实运行命令。

"$EXEC" \
    --data "$BASE_DATA" \
    --query "$QUERY_DATA" \
    --topK 10 \
    --ef 100 \
    --T 10 \
    > "$OUTPUT_FILE"

echo "======================================"
echo " Demo finished!"
echo " Result saved to:"
echo " $OUTPUT_FILE"
echo "======================================"

echo ""
echo "Preview result:"
head -n 30 "$OUTPUT_FILE"