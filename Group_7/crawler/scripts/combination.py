import os
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = PROJECT_ROOT / "5506" / "cleaned"

def merge_csv_files(folder_path, output_filename):
    """
    合并指定文件夹中的所有CSV文件
    
    Args:
        folder_path: 包含CSV文件的文件夹路径
        output_filename: 输出文件名（不包含路径）
    """
    folder = Path(folder_path)
    
    # 获取所有CSV文件
    csv_files = list(folder.glob('*.csv'))
    
    if not csv_files:
        print(f"在 {folder_path} 中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 读取并合并所有CSV文件
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            print(f"读取文件: {file.name} - 行数: {len(df)}")
            dataframes.append(df)
        except Exception as e:
            print(f"读取文件 {file.name} 时出错: {e}")
    
    if dataframes:
        # 合并所有数据
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        # 保存合并后的文件
        output_path = folder / output_filename
        merged_df.to_csv(output_path, index=False)
        print(f"合并完成！总共 {len(merged_df)} 行数据")
        print(f"输出文件: {output_path}")
        return len(merged_df)
    else:
        print("没有成功读取任何文件")
        return 0

def main():
    # 定义两个文件夹的路径
    features_folder = CLEANED_DIR / "features"
    labels_folder = CLEANED_DIR / "labels"
    
    print("=== 开始合并 features 文件夹 ===")
    features_count = merge_csv_files(features_folder, "all_features.csv")
    
    print("\n=== 开始合并 labels 文件夹 ===")
    labels_count = merge_csv_files(labels_folder, "all_labels.csv")
    
    print(f"\n=== 合并统计 ===")
    print(f"Features 合并完成: {features_count} 行数据")
    print(f"Labels 合并完成: {labels_count} 行数据")
    print("所有文件已生成到对应文件夹中！")

if __name__ == "__main__":
    main()
