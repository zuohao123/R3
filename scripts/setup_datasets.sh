#!/bin/bash

# R³ 数据集下载和设置脚本
# 使用方法: ./setup_datasets.sh [dataset_name]
# 如果不指定数据集名称，将显示所有可用数据集的设置说明

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data_pipeline/data"

echo -e "${BLUE}R³ 数据集设置工具${NC}"
echo "=================================="

# 检查数据目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}错误: 数据目录不存在: $DATA_DIR${NC}"
    exit 1
fi

# 显示数据集状态
show_dataset_status() {
    local dataset=$1
    local dataset_dir="$DATA_DIR/$dataset"
    
    if [ ! -d "$dataset_dir" ]; then
        echo -e "${RED}❌ $dataset: 目录不存在${NC}"
        return
    fi
    
    # 检查标注文件
    local train_file="$dataset_dir/${dataset}_train.json"
    local val_file="$dataset_dir/${dataset}_val.json"
    
    # 检查图像目录
    local img_dir=""
    case $dataset in
        "chartqa")
            img_dir="$dataset_dir/charts"
            ;;
        "docvqa")
            img_dir="$dataset_dir/documents"
            ;;
        *)
            img_dir="$dataset_dir/images"
            ;;
    esac
    
    local status="✅"
    local details=""
    
    if [ ! -f "$train_file" ]; then
        status="⚠️"
        details="${details} 缺少训练集标注"
    fi
    
    if [ ! -f "$val_file" ]; then
        status="⚠️"
        details="${details} 缺少验证集标注"
    fi
    
    if [ ! -d "$img_dir" ] || [ -z "$(ls -A "$img_dir" 2>/dev/null)" ]; then
        status="⚠️"
        details="${details} 缺少图像文件"
    fi
    
    if [ "$status" = "✅" ]; then
        local img_count=$(find "$img_dir" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)
        details=" (${img_count} 张图像)"
    fi
    
    echo -e "$status $dataset$details"
}

# 显示所有数据集状态
show_all_status() {
    echo -e "${YELLOW}数据集状态:${NC}"
    echo "----------"
    show_dataset_status "textvqa"
    show_dataset_status "mp_docvqa"
    show_dataset_status "infovqa"
    show_dataset_status "chartqa"
    show_dataset_status "docvqa"
    show_dataset_status "slidevqa"
    echo ""
}

# 显示数据集设置说明
show_dataset_info() {
    local dataset=$1
    local dataset_dir="$DATA_DIR/$dataset"
    
    echo -e "${GREEN}$dataset 数据集设置说明:${NC}"
    echo "------------------------"
    echo "目录位置: $dataset_dir"
    echo ""
    
    case $dataset in
        "textvqa")
            echo "官方网站: https://textvqa.org/"
            echo "下载步骤:"
            echo "1. 下载标注文件并重命名为:"
            echo "   - textvqa_train.json"
            echo "   - textvqa_val.json"
            echo "2. 下载图像文件到 images/ 目录"
            ;;
        "mp_docvqa")
            echo "官方网站: https://rrc.cvc.uab.es/?ch=17"
            echo "下载步骤:"
            echo "1. 下载标注文件并重命名为:"
            echo "   - mp_docvqa_train.json"
            echo "   - mp_docvqa_val.json"
            echo "2. 下载文档页面图像到 images/ 目录"
            ;;
        "infovqa")
            echo "官方网站: https://www.docvqa.org/datasets/infographicvqa"
            echo "下载步骤:"
            echo "1. 下载标注文件并重命名为:"
            echo "   - infovqa_train.json"
            echo "   - infovqa_val.json"
            echo "2. 下载信息图表图像到 images/ 目录"
            ;;
        "chartqa")
            echo "官方网站: https://github.com/vis-nlp/ChartQA"
            echo "下载步骤:"
            echo "1. 下载标注文件并重命名为:"
            echo "   - chartqa_train.json"
            echo "   - chartqa_val.json"
            echo "2. 下载图表图像到 charts/ 目录 (注意目录名)"
            ;;
        "docvqa")
            echo "官方网站: https://www.docvqa.org/"
            echo "下载步骤:"
            echo "1. 下载标注文件并重命名为:"
            echo "   - docvqa_train.json"
            echo "   - docvqa_val.json"
            echo "2. 下载文档图像到 documents/ 目录 (注意目录名)"
            ;;
        "slidevqa")
            echo "数据来源: 根据具体来源下载"
            echo "下载步骤:"
            echo "1. 下载标注文件并重命名为:"
            echo "   - slidevqa_train.json"
            echo "   - slidevqa_val.json"
            echo "2. 下载幻灯片图像到 images/ 目录"
            ;;
    esac
    
    echo ""
    echo "详细说明请查看: $dataset_dir/README.md"
    echo ""
}

# 验证数据集
validate_dataset() {
    local dataset=$1
    echo -e "${YELLOW}验证 $dataset 数据集...${NC}"
    
    cd "$PROJECT_ROOT"
    python -c "
try:
    from data_pipeline.datasets.$dataset import ${dataset^}Dataset
    from pathlib import Path
    dataset = ${dataset^}Dataset(Path('data_pipeline/data/$dataset'), 'train')
    print(f'✅ $dataset 数据集验证成功: {len(dataset)} 个样本')
    if len(dataset) > 0:
        sample = dataset[0]
        print(f'   示例ID: {sample[\"id\"]}')
        print(f'   问题: {sample[\"question\"][:50]}...')
except Exception as e:
    print(f'❌ $dataset 数据集验证失败: {e}')
"
}

# 主函数
main() {
    if [ $# -eq 0 ]; then
        # 显示所有数据集状态和说明
        show_all_status
        echo -e "${BLUE}使用方法:${NC}"
        echo "./setup_datasets.sh [dataset_name]  # 显示特定数据集的设置说明"
        echo "./setup_datasets.sh validate [dataset_name]  # 验证数据集"
        echo ""
        echo -e "${BLUE}可用数据集:${NC}"
        echo "- textvqa"
        echo "- mp_docvqa"
        echo "- infovqa"
        echo "- chartqa"
        echo "- docvqa"
        echo "- slidevqa"
        echo ""
        echo -e "${YELLOW}提示: 查看各数据集目录下的 README.md 文件获取详细说明${NC}"
        
    elif [ "$1" = "validate" ]; then
        if [ $# -eq 1 ]; then
            # 验证所有数据集
            echo -e "${YELLOW}验证所有数据集...${NC}"
            for dataset in textvqa mp_docvqa infovqa chartqa docvqa slidevqa; do
                validate_dataset "$dataset"
            done
        else
            # 验证特定数据集
            validate_dataset "$2"
        fi
        
    else
        # 显示特定数据集的设置说明
        local dataset=$1
        case $dataset in
            "textvqa"|"mp_docvqa"|"infovqa"|"chartqa"|"docvqa"|"slidevqa")
                show_dataset_info "$dataset"
                show_dataset_status "$dataset"
                ;;
            *)
                echo -e "${RED}错误: 未知的数据集 '$dataset'${NC}"
                echo "可用数据集: textvqa, mp_docvqa, infovqa, chartqa, docvqa, slidevqa"
                exit 1
                ;;
        esac
    fi
}

main "$@"