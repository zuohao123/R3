"""
MP-DocVQA 数据格式转换工具

将原始的 MP-DocVQA 格式转换为适合 R³ 框架的格式
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def convert_mp_docvqa_format(input_file: Path, output_file: Path) -> None:
    """
    转换 MP-DocVQA 数据格式
    
    输入格式:
    {
        "questionId": 337,
        "question": "what is the date mentioned in this letter?",
        "doc_id": "xnbl0037", 
        "page_ids": ["xnbl0037_p0", "xnbl0037_p1"],
        "answers": ["1/8/93"],
        "answer_page_idx": 0,
        "data_split": "train"
    }
    
    输出格式: 保持原格式，但确保兼容性
    """
    
    print(f"转换 {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("输入数据应该是一个列表")
    
    converted_data = []
    
    for item in data:
        # 验证必需字段
        required_fields = ["questionId", "question", "doc_id", "page_ids"]
        for field in required_fields:
            if field not in item:
                print(f"警告: 缺少必需字段 '{field}' 在问题 {item.get('questionId', 'unknown')}")
                continue
        
        # 标准化数据格式
        converted_item = {
            "questionId": item["questionId"],
            "question": item["question"],
            "doc_id": item["doc_id"],
            "page_ids": item["page_ids"],
            "answers": item.get("answers", []),
            "answer_page_idx": item.get("answer_page_idx", 0),
            "data_split": item.get("data_split", "train")
        }
        
        # 添加可选字段
        if "ocr_tokens" in item:
            converted_item["ocr_tokens"] = item["ocr_tokens"]
        if "captions" in item:
            converted_item["captions"] = item["captions"]
        
        converted_data.append(converted_item)
    
    # 保存转换后的数据
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成: {len(converted_data)} 个问题")


def validate_mp_docvqa_format(file_path: Path) -> bool:
    """验证 MP-DocVQA 数据格式"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("❌ 数据应该是一个列表")
            return False
        
        required_fields = ["questionId", "question", "doc_id", "page_ids"]
        
        for i, item in enumerate(data[:5]):  # 检查前5个样本
            for field in required_fields:
                if field not in item:
                    print(f"❌ 样本 {i} 缺少字段: {field}")
                    return False
            
            if not isinstance(item["page_ids"], list):
                print(f"❌ 样本 {i} 的 page_ids 应该是列表")
                return False
            
            if len(item["page_ids"]) == 0:
                print(f"❌ 样本 {i} 的 page_ids 不能为空")
                return False
        
        print(f"✅ 格式验证通过: {len(data)} 个问题")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="MP-DocVQA 数据格式转换工具")
    parser.add_argument("--input", type=Path, required=True, help="输入文件路径")
    parser.add_argument("--output", type=Path, help="输出文件路径")
    parser.add_argument("--validate", action="store_true", help="只验证格式，不转换")
    
    args = parser.parse_args()
    
    if args.validate:
        validate_mp_docvqa_format(args.input)
    else:
        if not args.output:
            # 默认输出路径
            args.output = args.input.parent / f"mp_docvqa_{args.input.stem}.json"
        
        convert_mp_docvqa_format(args.input, args.output)
        
        # 验证转换后的文件
        print("\n验证转换后的文件:")
        validate_mp_docvqa_format(args.output)


if __name__ == "__main__":
    main()