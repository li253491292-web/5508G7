import pandas as pd
import re
import json
import time
import os
import glob
from pathlib import Path
import requests
import sys
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from .llm_config import resolve_llm_config

# --- 1. 基础配置 ---
LLM_PROVIDER, API_KEY, URL, LLM_MODEL = resolve_llm_config()

# 你的路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = PROJECT_ROOT / "5506" / "cleaned"
INPUT_DIR = CLEANED_DIR / "cleaned"
OUTPUT_DIR = CLEANED_DIR / "features"

# 并发配置：如果你发现 Error 较多，请将 MAX_WORKERS 改小（如 2 或 3）
MAX_WORKERS = 2 
lock = Lock()

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'post_feature_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_llm_analysis(row_data):
    """
    【最核心的完整 PROMPT 落地】
    增强版LLM分析函数，包含完善的错误处理
    """
    # 完整的系统提示词定义
    full_system_prompt = """你是一位精通计算传播学、社会心理学与海外医疗营销的资深分析师。
你的任务是分析 Reddit 帖子内容，并根据医疗推演需求提取深度语义标签。

请严格遵守以下【字段定义】进行标注，并仅返回 JSON 格式：

1. content_angle (内容视角): 
   - Personal Experience (个人亲历分享)
   - Question/Inquiry (向网友求助/咨询)
   - Recommendation (明确推荐某地或某医生)
   - Critical/Skeptical (质疑中医疗效或中国医疗环境)
   - Informational (纯转发资讯、新闻或政策)
   - Off-topic (与医疗健康完全无关的内容，如游戏、玩笑)

2. primary_topic (核心主题): 
   - Clinical Efficacy (疗效/治愈率)
   - Cost/Price (治疗、机票、食宿费用)
   - Safety/Regulation (合法性、安全、医疗事故风险)
   - Education/Training (医生资质、中医师承、学历)
   - Travel Logistics (签证、翻译、医院沟通、交通)

3. secondary_topics: 帖子涉及的其他次要话题列表 (如: ['Acupuncture', 'Beijing', 'Herbal Tea'])

4. motivation_type (发帖动机): 
   - Chronic Pain (寻找慢性病解决方案)
   - Wellness/Biohacking (追求长寿、优化体能)
   - Career Development (想来华学习中医并以此为业)
   - Academic/Policy (研究中国医疗政策或中医文化)

5. pain_point: 用户最核心的痛点描述 (30字以内，如: "本地西医无法治愈"、"翻译费用过高")

6. decision_stage (决策阶段): 
   - Discovery (初步了解，持好奇态度)
   - Evaluation (正在对比不同医院或在犹豫风险)
   - Commitment (已准备出发、正在治疗或已完成治疗)

7. risk_sensitivity: 风险敏感度 (High/Medium/Low)

8. trust_signal_focus (信任信号点): 用户最在乎什么来建立信任？ (Doctor Background: 医生资历; Scientific Evidence: 论文数据; Peer Anecdotes: 网友评价; Institution Rank: 医院等级)

9. sentiment: 情感倾向 (Positive / Neutral / Negative)

10. emotional_intensity: 情绪强度 (1-5 评分，5为最激烈)

11. commercial_intent: 商业意图强度 (1-5 评分，识别是否为医疗中介或软广)

12. information_gap_level: 信息缺口程度 (1-5 评分，分数越高代表用户对真实情况越无知)

13. persona_hint: 30字内人格画像简述 (如: "对东方医学好奇的美国中产"、"严谨的科学怀疑论者")

14. approval_bucket (态度分桶): (Supportive: 支持中医/中国; Skeptical: 怀疑但观望; Opposed: 坚决反对)
15. has_url_in_body: 布尔值 (true/false)。判断帖子正文是否包含 URL 链接。
16. has_question_mark: 布尔值 (true/false)。判断标题或正文是否包含问号。
17. has_exclamation_mark: 布尔值 (true/false)。判断标题或正文是否包含感叹号。

【输出要求】:
- 必须返回纯 JSON，不能包含 Markdown 标签或任何解释文字。
- 必须确保字段名称与上述要求完全一致。
- 布尔值字段请统一使用标准 JSON 布尔格式 (true 或 false)。"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 构造请求数据
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": f"Title: {row_data['title']}\nContent: {row_data['selftext']}"}
        ],
        "response_format": {"type": "json_object"} # 强制 Qwen 返回 JSON
    }

    for attempt in range(3): # 失败重试
        try:
            res = requests.post(URL, headers=headers, json=payload, timeout=60)
            if res.status_code == 200:
                content = res.json()['choices'][0]['message']['content']
                result = json.loads(content)
                
                # 验证返回的数据结构
                required_fields = ["content_angle", "primary_topic", "decision_stage"]
                missing_fields = [f for f in required_fields if f not in result]
                
                if missing_fields:
                    logging.warning(f"LLM返回缺少字段：{missing_fields}")
                    # 用默认值填充缺失字段
                    for field in missing_fields:
                        result[field] = "Unknown"
                
                return result
                
            elif res.status_code == 429:
                wait_time = 5 * (attempt + 1)
                logging.warning(f"API限流，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)

            elif res.status_code == 400:
                # 请求内容本身有问题，直接记录并跳过，不再重试
                logging.error(f"API 400 错误，跳过该条记录：post_id={row_data.get('post_id')}，文本={row_data.get('title','')[:60]}...，响应={res.text}")
                return {
                    "content_angle": "Error",
                    "primary_topic": "Error",
                    "decision_stage": "Error",
                    "error_reason": f"400 Bad Request: {res.text}"
                }

            else:
                logging.error(f"API调用失败，状态码：{res.status_code}，响应：{res.text}")
                
        except json.JSONDecodeError as e:
            logging.error(f"JSON解析失败：{str(e)}")
            logging.error(f"原始内容：{content[:200]}...")
            
        except requests.exceptions.Timeout:
            logging.warning(f"请求超时，第 {attempt + 1} 次重试")
            
        except Exception as e:
            logging.error(f"API调用异常：{str(e)}")
            
    # 所有重试都失败，返回错误信息
    logging.error(f"分析帖子失败：{row_data.get('title', 'Unknown')[:50]}...")
    return {
        "content_angle": "Error",
        "primary_topic": "Error", 
        "decision_stage": "Error",
        "error_reason": "LLM分析失败"
    }

def extract_keyword(file_name):
    """提取关键词的辅助函数"""
    keyword = (file_name.replace("post_final_", "")
               .replace(".csv", "")
               .replace("_cleaned", "")
               .replace("post_", ""))
    return keyword or "unknown"

def preprocess_dataframe(df):
    """数据预处理"""
    # 确保必要字段存在
    required_cols = ['title', 'selftext', 'post_id']
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        logging.warning(f"缺少字段 {missing_cols}，将用空值填充")
        for col in missing_cols:
            df[col] = ""
    
    # 物理计算字段
    df['full_text'] = df['title'].fillna('') + "\n" + df['selftext'].fillna('')
    df['title_length'] = df['title'].str.len()
    df['selftext_length'] = df['selftext'].fillna('').str.len()
    
    # 计算参与度得分，确保字段存在
    score_col = 'score' if 'score' in df.columns else None
    comments_col = 'num_comments' if 'num_comments' in df.columns else None
    
    if score_col and comments_col:
        df['engagement_score'] = df[score_col].fillna(0) + (df[comments_col].fillna(0) * 2)
    elif score_col:
        df['engagement_score'] = df[score_col].fillna(0)
    elif comments_col:
        df['engagement_score'] = df[comments_col].fillna(0) * 2
    else:
        df['engagement_score'] = 0
    
    return df

def get_processed_ids(output_path):
    """获取已处理的ID列表"""
    if not os.path.exists(output_path):
        return set()

    # 优先使用 pandas；读取失败时降级到逐行解析，防止小干扰导致整个流程中断
    try:
        processed_df = pd.read_csv(output_path, dtype=str, on_bad_lines='skip', engine='python')
        if 'post_id' in processed_df.columns:
            return set(processed_df['post_id'].astype(str).unique())
        else:
            logging.warning("已处理文件中未找到 post_id 列，尝试逐行解析")
    except Exception as e:
        logging.warning(f"读取已处理文件失败: {str(e)}")

    try:
        import csv
        ids = set()
        with open(output_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'post_id' in row and row['post_id'] is not None:
                    ids.add(str(row['post_id']))
        return ids
    except Exception as e:
        logging.warning(f"逐行解析已处理文件失败: {str(e)}")
        return set()

def filter_unprocessed_records(df, processed_ids):
    """过滤未处理的记录"""
    return df[~df['post_id'].astype(str).isin(processed_ids)].to_dict('records')

def save_results_batch(results, output_path):
    """批量保存结果"""
    try:
        is_new = not os.path.exists(output_path)
        pd.DataFrame(results).to_csv(
            output_path, 
            mode='a', 
            index=False, 
            header=is_new, 
            encoding='utf-8-sig'
        )
    except Exception as e:
        logging.error(f"保存结果失败: {str(e)}")

def process_file(file_path):
    """增强版文件处理函数"""
    file_name = os.path.basename(file_path)
    keyword = extract_keyword(file_name)
    output_path = os.path.join(OUTPUT_DIR, f"post_feature_{keyword}.csv")
    
    logging.info(f"开始处理文件：{file_name} (关键词: {keyword})")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        logging.info(f"成功加载 {len(df)} 条记录")
        
        # 数据预处理
        df = preprocess_dataframe(df)
        
        # 断点续传检查
        processed_ids = get_processed_ids(output_path)
        todo_list = filter_unprocessed_records(df, processed_ids)
        
        if not todo_list:
            logging.info(f"文件 {file_name} 已完成处理，跳过")
            return
            
        logging.info(f"需要处理 {len(todo_list)} 条新记录")
        
        # 并发处理
        process_with_threadpool(todo_list, output_path, keyword)
        
        logging.info(f"文件 {file_name} 处理完成")
        
    except pd.errors.EmptyDataError:
        logging.error(f"文件 {file_name} 为空文件")
    except pd.errors.ParserError as e:
        logging.error(f"CSV解析失败 {file_name}: {str(e)}")
    except Exception as e:
        logging.error(f"处理文件 {file_name} 时发生未知错误: {str(e)}")

def process_with_threadpool(todo_list, output_path, keyword):
    """使用线程池处理数据"""
    total = len(todo_list)
    completed = 0
    results_buffer = []
    
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            future_to_row = {
                executor.submit(get_llm_analysis, row): row 
                for row in todo_list
            }
            
            for future in as_completed(future_to_row):
                row_data = future_to_row[future]
                
                try:
                    llm_result = future.result(timeout=30)  # 设置超时
                    combined_row = {**row_data, **llm_result}
                    results_buffer.append(combined_row)
                    completed += 1
                    
                    # 实时保存结果
                    if len(results_buffer) >= 5:  # 每5条保存一次
                        save_results_batch(results_buffer, output_path)
                        results_buffer = []

                    # 显示进度（每条都打印，避免误判“卡住”）
                    progress = (completed / total) * 100
                    logging.info(
                        f"[{keyword}] 进度: {progress:.1f}% ({completed}/{total}) | "
                        f"当前: {row_data.get('title', '')[:30]}..."
                    )
                        
                except TimeoutError:
                    logging.warning(f"处理超时，跳过记录: {row_data.get('post_id')}")
                except Exception as e:
                    logging.error(f"处理记录失败 {row_data.get('post_id')}: {str(e)}")
    except KeyboardInterrupt:
        logging.warning("用户中断处理，正在保存已完成的数据并退出")
        if results_buffer:
            save_results_batch(results_buffer, output_path)
        raise
    except Exception as e:
        logging.error(f"线程池处理出现异常: {str(e)}")
        if results_buffer:
            save_results_batch(results_buffer, output_path)
        raise
    finally:
        if results_buffer:
            save_results_batch(results_buffer, output_path)

    logging.info(f"[{keyword}] 处理完成，共处理 {completed} 条记录")

def run():
    """主运行函数，增强版"""
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*post*.csv"))
    
    # 添加空列表检查
    if not csv_files:
        logging.warning("警告：未找到任何CSV文件，请检查输入目录路径是否正确")
        logging.warning(f"当前输入目录：{INPUT_DIR}")
        return
    
    logging.info(f"📁 发现 {len(csv_files)} 个关键词文件")
    
    # 按文件名排序，确保处理顺序一致
    csv_files.sort()
    
    # 统计信息
    total_files = len(csv_files)
    success_files = 0
    failed_files = 0
    
    for i, f in enumerate(csv_files, 1):
        try:
            logging.info(f"\n{'='*50}")
            logging.info(f"处理文件 [{i}/{total_files}]: {os.path.basename(f)}")
            process_file(f)
            success_files += 1
            
        except Exception as e:
            logging.error(f"❌ 处理文件 {f} 失败：{str(e)}")
            failed_files += 1
            continue
    
    # 最终统计报告
    logging.info(f"\n{'='*50}")
    logging.info(f"📊 处理完成统计:")
    logging.info(f"   总文件数: {total_files}")
    logging.info(f"   成功处理: {success_files}")
    logging.info(f"   失败文件: {failed_files}")
    logging.info(f"   成功率: {(success_files/total_files)*100:.1f}%")

def validate_config():
    """验证配置"""
    if not os.path.exists(INPUT_DIR):
        raise ValueError(f"输入目录不存在: {INPUT_DIR}")
    
    if not API_KEY:
        raise ValueError("API密钥未设置")
    
    logging.info("✅ 配置验证通过")

if __name__ == "__main__":
    try:
        # 验证配置
        validate_config()
        
        # 开始处理
        logging.info("🚀 开始 Reddit 帖子特征分析...")
        start_time = datetime.now()
        
        run()
        
        # 处理完成
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"\n🎉 所有任务处理完成！总耗时: {duration}")
        
    except KeyboardInterrupt:
        logging.warning("用户中断处理")
    except Exception as e:
        logging.error(f"程序运行失败: {str(e)}")
        raise
