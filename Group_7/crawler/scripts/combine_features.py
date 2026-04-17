import pandas as pd
import os
import logging
from pathlib import Path
from typing import List, Optional, Union
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = PROJECT_ROOT / "5506" / "cleaned"
INPUT_DIR = CLEANED_DIR / "features"
OUTPUT_DIR = CLEANED_DIR / "features"
OUTPUT_NAME = "post_feature"

def read_csv_file(file_path: Union[str, Path], validate: bool = True) -> Optional[pd.DataFrame]:
    """读取单个CSV文件并进行基础验证"""
    try:
        df = pd.read_csv(file_path)
        
        if validate and df.empty:
            logging.warning(f"文件为空: {file_path}")
            return None
            
        # 基础数据质量检查
        if validate:
            # 检查是否有重复的列名
            if df.columns.duplicated().any():
                logging.warning(f"文件 {file_path} 包含重复列名")
                
            # 检查是否有空列
            empty_cols = df.columns[df.isnull().all()].tolist()
            if empty_cols:
                logging.warning(f"文件 {file_path} 包含空列: {empty_cols}")
        
        return df
    except pd.errors.EmptyDataError:
        logging.warning(f"文件为空: {file_path}")
        return None
    except pd.errors.ParserError as e:
        logging.error(f"CSV解析错误 {file_path}: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"无法读取 {file_path}: {type(e).__name__}: {str(e)}")
        return None

def combine_csv_files_basic(
    input_dir: Union[str, Path], 
    output_dir: Union[str, Path], 
    output_name: str,
    validate_files: bool = True
) -> Optional[pd.DataFrame]:
    """
    基础顺序处理CSV文件合并
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径  
        output_name: 输出文件名（不含扩展名）
        validate_files: 是否验证文件质量
        
    Returns:
        合并后的DataFrame，如果失败返回None
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logging.error(f"输入目录不存在: {input_path}")
        return None
        
    # 获取所有 CSV 文件并排序
    csv_files = sorted([f for f in input_path.glob("*.csv") if f.is_file()])
    
    if not csv_files:
        logging.error(f"未找到 CSV 文件在: {input_path}")
        return None
    
    logging.info(f"找到 {len(csv_files)} 个 CSV 文件，开始顺序拼接...")
    
    # 加载并拼接所有文件
    df_list = []
    for i, file_path in enumerate(csv_files, 1):
        df = read_csv_file(file_path, validate_files)
        if df is not None:
            df_list.append(df)
            logging.info(f"[{i}/{len(csv_files)}] 加载完成: {file_path.name} ({len(df)} 行)")
        else:
            logging.warning(f"[{i}/{len(csv_files)}] 跳过无效文件: {file_path.name}")
    
    if not df_list:
        logging.error("没有成功读取任何有效文件")
        return None
    
    # 合并所有数据框
    logging.info("正在合并数据框...")
    combined_df = pd.concat(df_list, ignore_index=True, sort=False)
    
    return combined_df
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logging.error(f"输入目录不存在: {input_path}")
        return None
        
    # 获取所有 CSV 文件并排序
    csv_files = sorted([f for f in input_path.glob("*.csv") if f.is_file()])
    
    if not csv_files:
        logging.error(f"未找到 CSV 文件在: {input_path}")
        return None
    
    logging.info(f"找到 {len(csv_files)} 个 CSV 文件，开始顺序拼接...")
    
    # 加载并拼接所有文件
    df_list = []
    for i, file_path in enumerate(csv_files, 1):
        df = read_csv_file(file_path, validate_files)
        if df is not None:
            df_list.append(df)
            logging.info(f"[{i}/{len(csv_files)}] 加载完成: {file_path.name} ({len(df)} 行)")
        else:
            logging.warning(f"[{i}/{len(csv_files)}] 跳过无效文件: {file_path.name}")
    
    if not df_list:
        logging.error("没有成功读取任何有效文件")
        return None
    
    # 合并所有数据框
    logging.info("正在合并数据框...")
    combined_df = pd.concat(df_list, ignore_index=True, sort=False)
    
    return combined_df

def save_multiple_formats(df: pd.DataFrame, output_path: Path, output_name: str) -> None:
    """
    保存DataFrame为多种格式（CSV、Parquet、HDF5）
    
    Args:
        df: 要保存的DataFrame
        output_path: 输出目录路径
        output_name: 输出文件名（不含扩展名）
    """
    
    # ========== 三种输出格式 ==========
    
    # 1. CSV 格式（兼容性最好，但文件大）
    csv_path = output_path / f"{output_name}.csv"
    try:
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        csv_size = csv_path.stat().st_size / (1024**2)  # 转换为 MB
        logging.info(f"📄 CSV 已保存: {csv_path} ({csv_size:.2f} MB)")
    except Exception as e:
        logging.error(f"CSV 保存失败: {str(e)}")
    
    # 2. Parquet 格式（推荐：快速、高效压缩）
    parquet_path = output_path / f"{output_name}.parquet"
    try:
        df.to_parquet(parquet_path, index=False, compression='snappy')
        parquet_size = parquet_path.stat().st_size / (1024**2)
        logging.info(f"⚡ Parquet 已保存: {parquet_path} ({parquet_size:.2f} MB)")
    except Exception as e:
        logging.warning(f"Parquet 保存失败: {str(e)}")
    
    # 3. HDF5 格式（适合超大数据）
    hdf5_path = output_path / f"{output_name}.h5"
    try:
        df.to_hdf(hdf5_path, key='data', mode='w', complevel=9)
        hdf5_size = hdf5_path.stat().st_size / (1024**2)
        logging.info(f"💾 HDF5 已保存: {hdf5_path} ({hdf5_size:.2f} MB)")
    except Exception as e:
        logging.warning(f"HDF5 保存失败: {str(e)}")

def combine_csv_files_parallel(
    input_dir: Union[str, Path], 
    output_dir: Union[str, Path], 
    output_name: str,
    max_workers: Optional[int] = None,
    chunk_size: int = 10000,
    validate_files: bool = True
) -> Optional[pd.DataFrame]:
    """
    并行拼接目录下所有 CSV 文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径  
        output_name: 输出文件名（不含扩展名）
        max_workers: 最大并行工作进程数，None表示使用CPU核心数
        chunk_size: 大文件分块读取大小
        validate_files: 是否验证文件质量
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logging.error(f"输入目录不存在: {input_path}")
        return None
        
    # 获取所有 CSV 文件并排序
    csv_files = sorted([f for f in input_path.glob("*.csv") if f.is_file()])
    
    if not csv_files:
        logging.error(f"未找到 CSV 文件在: {input_path}")
        return None
    
    logging.info(f"找到 {len(csv_files)} 个 CSV 文件，开始并行拼接...")
    
    # 使用进程池并行读取文件
    df_list = []
    max_workers = max_workers or mp.cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有读取任务
        future_to_file = {
            executor.submit(read_csv_file, file_path, validate_files): file_path 
            for file_path in csv_files
        }
        
        # 收集结果
        for i, future in enumerate(as_completed(future_to_file), 1):
            file_path = future_to_file[future]
            try:
                df = future.result()
                if df is not None:
                    df_list.append(df)
                    logging.info(f"[{i}/{len(csv_files)}] 加载完成: {file_path.name} ({len(df)} 行)")
                else:
                    logging.warning(f"[{i}/{len(csv_files)}] 跳过无效文件: {file_path.name}")
            except Exception as e:
                logging.error(f"处理文件 {file_path.name} 时出错: {str(e)}")
    
    if not df_list:
        logging.error("没有成功读取任何有效文件")
        return None
    
    # 合并所有数据框
    logging.info("正在合并数据框...")
    combined_df = pd.concat(df_list, ignore_index=True, sort=False)
    logging.info(f"✅ 拼接完成，总计 {len(combined_df)} 行，{len(combined_df.columns)} 列")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 三种输出格式 ==========
    
    # 1. CSV 格式（兼容性最好，但文件大）
    csv_path = os.path.join(output_dir, f"{output_name}.csv")
    combined_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    csv_size = os.path.getsize(csv_path) / (1024**2)  # 转换为 MB
    logging.info(f"📄 CSV 已保存: {csv_path} ({csv_size:.2f} MB)")
    
    # 2. Parquet 格式（推荐：快速、高效压缩）
    try:
        parquet_path = os.path.join(output_dir, f"{output_name}.parquet")
        combined_df.to_parquet(parquet_path, index=False, compression='snappy')
        parquet_size = os.path.getsize(parquet_path) / (1024**2)
        logging.info(f"⚡ Parquet 已保存: {parquet_path} ({parquet_size:.2f} MB)")
    except Exception as e:
        logging.warning(f"Parquet 保存失败: {str(e)}")
    
    # 3. HDF5 格式（适合超大数据）
    try:
        hdf5_path = os.path.join(output_dir, f"{output_name}.h5")
        combined_df.to_hdf(hdf5_path, key='data', mode='w', complevel=9)
        hdf5_size = os.path.getsize(hdf5_path) / (1024**2)
        logging.info(f"💾 HDF5 已保存: {hdf5_path} ({hdf5_size:.2f} MB)")
    except Exception as e:
        logging.warning(f"HDF5 保存失败: {str(e)}")
    
    # 打印数据概览
    logging.info(f"\n📊 数据概览:")
    logging.info(f"   行数: {len(combined_df)}")
    logging.info(f"   列数: {len(combined_df.columns)}")
    logging.info(f"   列名: {list(combined_df.columns)[:5]}... (显示前5列)")
    logging.info(f"   内存占用: {combined_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

def combine_csv_files(
    input_dir: Union[str, Path], 
    output_dir: Union[str, Path], 
    output_name: str,
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
    validate_files: bool = True
) -> Optional[pd.DataFrame]:
    """
    拼接目录下所有 CSV 文件为一个大文件
    支持多种输出格式（CSV、Parquet、HDF5）
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        output_name: 输出文件名（不含扩展名）
        use_parallel: 是否使用并行处理
        max_workers: 最大并行工作进程数
        validate_files: 是否验证文件质量
        
    Returns:
        合并后的DataFrame，如果失败返回None
    """
    
    # 选择处理方式
    if use_parallel and len(list(Path(input_dir).glob("*.csv"))) > 5:  # 文件较多时使用并行
        combined_df = combine_csv_files_parallel(
            input_dir, output_dir, output_name, 
            max_workers, validate_files=validate_files
        )
    else:
        # 顺序处理（小文件或少量文件）
        combined_df = combine_csv_files_basic(input_dir, output_dir, output_name, validate_files)
    
    if combined_df is None:
        return None
    
    logging.info(f"✅ 拼接完成，总计 {len(combined_df)} 行，{len(combined_df.columns)} 列")
    
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存各种格式...
    save_multiple_formats(combined_df, output_path, output_name)
    
    # 内存清理
    gc.collect()
    
    return combined_df
