import pandas as pd
import re
import json
import time
import os
import glob
from pathlib import Path
import requests
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from .llm_config import resolve_llm_config

LLM_PROVIDER, API_KEY, URL, LLM_MODEL = resolve_llm_config()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = PROJECT_ROOT / "5506" / "cleaned"
INPUT_DIR = CLEANED_DIR / "cleaned"
OUTPUT_DIR = CLEANED_DIR / "labels"
COMMENT_INPUT_PATTERN = "comments_final_*_cleaned.csv"

MAX_WORKERS = 5
lock = Lock()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'comment_label_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_llm_analysis_comment(row_data):
    full_system_prompt = """你是一位精通Reddit评论分析和医学传播学的专家。
你的任务是分析Reddit的评论内容，并提取深度语义标签。

【字段定义与输出要求】
请严格返回以下JSON字段（布尔值用true/false，无markdown）：

1. stance - 单选: Supportive / Skeptical / Opposed / Neutral
2. intent - 单选: Advice / Question / Experience_Sharing / Correction / Recommendation / Support / Warning / Information
3. emotion - 单选: Positive / Neutral / Negative / Mixed
4. emotional_intensity - 数字1-5 (1=理性 5=极其激烈)
5. topic - 单选: Efficacy / Safety / Cost / Practitioner_Quality / Logistics / Personal_Testimony / Moderation / Off_Topic
6. trust_signal - 单选: Doctor_Background / Scientific_Evidence / Peer_Anecdotes / Institution_Rank / Personal_Authority / None
7. persona_signal - 字符串30字以内
8. relevance_to_post - 数字1-5 (1=无关 5=高度相关)
9. helpfulness_label - 单选: High / Medium / Low

【输出约束】
必须返回纯JSON，不能包含Markdown或解释文字。字段名必须完全匹配。布尔字段用true/false。数字字段返回整数1-5。字符串无多余空格。"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": f"Comment: {row_data.get('comment_body', '')}"}
        ],
        "response_format": {"type": "json_object"}
    }

    for attempt in range(3):
        try:
            res = requests.post(URL, headers=headers, json=payload, timeout=60)
            if res.status_code == 200:
                content = res.json()['choices'][0]['message']['content']
                result = json.loads(content)
                required_fields = ["stance", "intent", "emotion", "emotional_intensity", "topic", "trust_signal", "persona_signal", "relevance_to_post", "helpfulness_label"]
                for field in required_fields:
                    if field not in result:
                        if field in ["emotional_intensity", "relevance_to_post"]:
                            result[field] = 1
                        else:
                            result[field] = "Unknown"
                return result
            elif res.status_code == 429:
                wait_time = 5 * (attempt + 1)
                logging.warning(f"API限流，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                logging.error(f"API调用失败，状态码：{res.status_code}")
        except json.JSONDecodeError as e:
            logging.error(f"JSON解析失败：{str(e)}")
        except requests.exceptions.Timeout:
            logging.warning(f"请求超时，第 {attempt + 1} 次重试")
        except Exception as e:
            logging.error(f"API调用异常：{str(e)}")

    logging.error(f"分析评论失败：{row_data.get('comment_id', 'Unknown')}")
    return {"stance": "Unknown", "intent": "Unknown", "emotion": "Unknown", "emotional_intensity": 1, "topic": "Unknown", "trust_signal": "None", "persona_signal": "Unknown", "relevance_to_post": 1, "helpfulness_label": "Unknown"}

def derive_comment_features(row_data):
    comment_body = row_data.get('comment_body', '')
    parent_id = row_data.get('parent_id', '')
    
    comment_length = len(comment_body) if comment_body else 0
    is_top_level = parent_id.startswith('t3_') if parent_id else False
    contains_question = '?' in comment_body if comment_body else False
    
    advice_keywords = ['suggest', 'recommend', 'should', 'try', 'consider', 'advise', '建议', '可以试试']
    contains_advice = any(kw.lower() in comment_body.lower() for kw in advice_keywords) if comment_body else False
    
    exp_keywords = ['I had', 'my experience', 'I went', 'I tried', 'happened to me', 'I experienced', 'My first', '我的经验']
    contains_personal_experience = any(kw.lower() in comment_body.lower() for kw in exp_keywords) if comment_body else False
    
    correct_keywords = ['Actually', 'that\'s not', 'incorrect', 'false', 'wrong', 'you\'re mistaken', '不对', '错了']
    contains_correction = any(kw.lower() in comment_body.lower() for kw in correct_keywords) if comment_body else False
    
    recommend_keywords = ['recommend', 'I recommend', 'highly recommend', 'definitely go', '推荐']
    contains_recommendation = any(kw.lower() in comment_body.lower() for kw in recommend_keywords) if comment_body else False
    
    risk_keywords = ['risk', 'danger', 'harm', 'injury', 'nerve damage', 'adverse', '风险', '危险', '伤害', '副作用']
    contains_risk_concern = any(kw.lower() in comment_body.lower() for kw in risk_keywords) if comment_body else False
    
    cost_keywords = ['cost', 'price', 'expensive', 'afford', 'money', 'fee', '费用', '价格', '贵']
    contains_cost_concern = any(kw.lower() in comment_body.lower() for kw in cost_keywords) if comment_body else False
    
    efficacy_keywords = ['efficacy', 'effective', 'works', 'cure', 'healing', 'treatment', '疗效', '有效', '治疗']
    contains_efficacy_concern = any(kw.lower() in comment_body.lower() for kw in efficacy_keywords) if comment_body else False
    
    return {
        "comment_length": comment_length,
        "is_top_level": is_top_level,
        "contains_question": contains_question,
        "contains_advice": contains_advice,
        "contains_personal_experience": contains_personal_experience,
        "contains_correction": contains_correction,
        "contains_recommendation": contains_recommendation,
        "contains_risk_concern": contains_risk_concern,
        "contains_cost_concern": contains_cost_concern,
        "contains_efficacy_concern": contains_efficacy_concern
    }

def preprocess_dataframe(df):
    required_cols = ['comment_id', 'comment_body', 'post_id', 'parent_id']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logging.warning(f"缺少字段 {missing_cols}，将用空值填充")
        for col in missing_cols:
            df[col] = ""
    for col in required_cols:
        df[col] = df[col].fillna('')
    return df

def get_processed_ids(output_path):
    if not os.path.exists(output_path):
        return set()
    try:
        processed_df = pd.read_csv(output_path)
        return set(processed_df['comment_id'].astype(str).unique())
    except Exception as e:
        logging.warning(f"读取已处理文件失败: {str(e)}")
        return set()

def filter_unprocessed_records(df, processed_ids):
    return df[~df['comment_id'].astype(str).isin(processed_ids)].to_dict('records')

def save_results_batch(results, output_path, field_order=None):
    try:
        is_new = not os.path.exists(output_path)
        result_df = pd.DataFrame(results)
        if field_order:
            existing_cols = [c for c in field_order if c in result_df.columns]
            result_df = result_df[existing_cols]
        result_df.to_csv(output_path, mode='a', index=False, header=is_new, encoding='utf-8-sig')
    except Exception as e:
        logging.error(f"保存结果失败: {str(e)}")

def process_file(file_path):
    file_name = os.path.basename(file_path)
    keyword = file_name.replace("comments_final_", "").replace("_cleaned.csv", "")
    output_path = str(OUTPUT_DIR / f"comment_feature_{keyword}.csv")
    logging.info(f"开始处理文件：{file_name} (关键词: {keyword})")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"成功加载 {len(df)} 条评论记录")
        df = preprocess_dataframe(df)
        processed_ids = get_processed_ids(output_path)
        todo_list = filter_unprocessed_records(df, processed_ids)
        if not todo_list:
            logging.info(f"文件 {file_name} 已完成处理，跳过")
            return
        logging.info(f"需要处理 {len(todo_list)} 条新评论")
        process_with_threadpool(todo_list, output_path, keyword)
        logging.info(f"文件 {file_name} 处理完成")
    except pd.errors.EmptyDataError:
        logging.error(f"文件 {file_name} 为空文件")
    except pd.errors.ParserError as e:
        logging.error(f"CSV解析失败 {file_name}: {str(e)}")
    except Exception as e:
        logging.error(f"处理文件 {file_name} 时发生未知错误: {str(e)}")

def process_with_threadpool(todo_list, output_path, keyword):
    total = len(todo_list)
    completed = 0
    results_buffer = []
    field_order = ['comment_id', 'post_id', 'source_keyword', 'parent_id', 'subreddit', 'comment_body', 'comment_score', 'depth', 'created_utc', 'permalink', 'comment_length', 'is_top_level', 'contains_question', 'contains_advice', 'contains_personal_experience', 'contains_correction', 'contains_recommendation', 'contains_risk_concern', 'contains_cost_concern', 'contains_efficacy_concern', 'stance', 'intent', 'emotion', 'emotional_intensity', 'topic', 'trust_signal', 'persona_signal', 'relevance_to_post', 'helpfulness_label']
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {executor.submit(get_llm_analysis_comment, row): row for row in todo_list}
        for future in as_completed(future_to_row):
            row_data = future_to_row[future]
            try:
                llm_result = future.result(timeout=30)
                derived_features = derive_comment_features(row_data)
                combined_row = {**row_data, **derived_features, **llm_result}
                results_buffer.append(combined_row)
                completed += 1
                if len(results_buffer) >= 5:
                    save_results_batch(results_buffer, output_path, field_order)
                    results_buffer = []
                    progress = (completed / total) * 100
                    logging.info(f"[{keyword}] 进度: {progress:.1f}% ({completed}/{total})")
            except TimeoutError:
                logging.warning(f"处理超时，跳过评论: {row_data.get('comment_id')}")
            except Exception as e:
                logging.error(f"处理评论失败 {row_data.get('comment_id')}: {str(e)}")
    
    if results_buffer:
        save_results_batch(results_buffer, output_path, field_order)
    logging.info(f"[{keyword}] 处理完成，共处理 {completed} 条评论")

def run():
    csv_files = glob.glob(os.path.join(INPUT_DIR, COMMENT_INPUT_PATTERN))
    if not csv_files:
        logging.warning(f"⚠️ 未找到评论文件，当前输入目录：{INPUT_DIR}")
        return
    logging.info(f"📁 发现 {len(csv_files)} 个评论文件")
    csv_files.sort()
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
    logging.info(f"\n{'='*50}")
    logging.info(f"📊 处理完成统计:")
    logging.info(f"   总文件数: {total_files}")
    logging.info(f"   成功处理: {success_files}")
    logging.info(f"   失败文件: {failed_files}")
    if total_files > 0:
        logging.info(f"   成功率: {(success_files/total_files)*100:.1f}%")

def validate_config():
    if not os.path.exists(INPUT_DIR):
        raise ValueError(f"输入目录不存在: {INPUT_DIR}")
    if not API_KEY:
        raise ValueError("API密钥未设置")
    logging.info("✅ 配置验证通过")

if __name__ == "__main__":
    try:
        validate_config()
        logging.info("🚀 开始 Reddit 评论特征分析...")
        start_time = datetime.now()
        run()
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"\n🎉 所有任务处理完成！总耗时: {duration}")
    except KeyboardInterrupt:
        logging.warning("\n⚠️ 用户中断处理")
    except Exception as e:
        logging.error(f"\n❌ 程序运行失败: {str(e)}")
        raise
