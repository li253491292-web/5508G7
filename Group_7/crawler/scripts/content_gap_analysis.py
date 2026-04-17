import os
import json
import logging
import time
import re
from pathlib import Path
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from .llm_config import resolve_llm_config

# ---------- 配置 ----------
LLM_PROVIDER, API_KEY, LLM_URL, LLM_MODEL = resolve_llm_config()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = PROJECT_ROOT / "5506" / "cleaned"
POST_COMMENT_AGG_PATH = CLEANED_DIR / "aggregation" / "post_comment_aggregate.csv"
POST_FEATURES_DIR = CLEANED_DIR / "features"
COMMENT_LABELS_DIR = CLEANED_DIR / "labels"
CONTENT_GAP_PATH = CLEANED_DIR / "content_gap" / "content_gap.csv"

# 入口参数
MIN_QUESTION_RATIO = 0.2
MIN_CORRECTION_RATIO = 0.1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_all_comment_labels(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("comments_labels_") and f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"在 {directory} 未找到评论标签文件")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def load_all_post_features(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("post_feature_") and f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"在 {directory} 未找到帖子特征文件")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def get_post_content(df_posts, post_id):
    if 'post_id' not in df_posts.columns:
        return '', ''
    row = df_posts[df_posts['post_id'] == post_id]
    if row.empty:
        return '', ''
    title = row.iloc[0].get('title', '') if 'title' in row.columns else ''
    selftext = row.iloc[0].get('selftext', '') if 'selftext' in row.columns else ''
    return str(title), str(selftext)


def get_sorted_comments_by_post(df_comments, post_id, top_n=5):
    subset = df_comments[df_comments['post_id'] == post_id]
    if subset.empty:
        return []
    if 'comment_score' in subset.columns:
        subset = subset.sort_values(by='comment_score', ascending=False)
    return subset['comment_body'].astype(str).fillna('').head(top_n).tolist()


def call_llm_for_gap_analysis(post_text, comments_text):
    prompt = f"""你是一位内容营销与医疗传播专家。\n\n帖子内容：\n{post_text}\n\n评论区代表性评论（最多5条）：\n{comments_text}\n\n请分析信息缺口，返回纯 JSON，包括字段：dominant_question, dominant_objection, missing_info_types, missing_trust_signals, missing_evidence_types, missing_decision_support, clarity_gap_summary, trust_gap_summary, risk_gap_summary, cost_gap_summary, efficacy_gap_summary, recommended_tone, recommended_structure, recommended_cta, avoid_points, suggested_content_fix。"""

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': LLM_MODEL,
        'messages': [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': '请返回 JSON 结构。'}
        ],
        'response_format': {'type': 'json_object'}
    }

    for i in range(3):
        try:
            r = requests.post(LLM_URL, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                content = r.json()['choices'][0]['message']['content']
                # 去掉 ```json 等可能冗余的标记
                content = content.strip()
                content = re.sub(r"^```(?:json)?", "", content, flags=re.IGNORECASE)
                content = re.sub(r"```$", "", content, flags=re.IGNORECASE)
                content = content.strip()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # 可能存在前缀文本，尝试寻找JSON子串
                    m = re.search(r"\{.*\}", content, flags=re.S)
                    if m:
                        return json.loads(m.group(0))
                    logging.warning('LLM返回内容无法解析为JSON，使用空值')
                    return {}
            if r.status_code == 429:
                logging.warning('LLM 限流，等待重试...')
                time.sleep(5 * (i + 1))
            else:
                logging.warning(f'LLM 返回状态 {r.status_code}')
        except Exception as e:
            logging.error(f'LLM 调用异常: {e}')
            time.sleep(2)

    return {}


def fill_gap_defaults(data):
    fields = [
        'dominant_question','dominant_objection','missing_info_types','missing_trust_signals',
        'missing_evidence_types','missing_decision_support','clarity_gap_summary','trust_gap_summary',
        'risk_gap_summary','cost_gap_summary','efficacy_gap_summary','recommended_tone',
        'recommended_structure','recommended_cta','avoid_points','suggested_content_fix'
    ]
    out = {}
    for f in fields:
        out[f] = data.get(f, 'Unknown') if isinstance(data, dict) else 'Unknown'
    return out


def derive_rewrite_and_goal(row):
    rw = 'Low'
    if row.get('question_ratio', 0) > 0.4 and row.get('total_comments', 0) > 10:
        rw = 'High'
    elif row.get('correction_ratio', 0) > 0.2:
        rw = 'High'
    elif row.get('question_ratio', 0) > 0.2 or row.get('correction_ratio', 0) > 0.1:
        rw = 'Medium'
    og = 'General Clarity'
    if row.get('risk_concern_ratio', 0) > row.get('cost_concern_ratio', 0):
        og = 'Safety Education'
    elif row.get('cost_concern_ratio', 0) > row.get('risk_concern_ratio', 0):
        og = 'Price Transparency'
    return rw, og


def process_candidate(row, df_comments, df_post_features):
    post_id = row['post_id']
    comments = get_sorted_comments_by_post(df_comments, post_id)
    comments_text = '\n---\n'.join(comments)

    title, selftext = get_post_content(df_post_features, post_id)
    if not title and not selftext:
        title = row.get('source_keyword', '')
        selftext = f"subreddit: {row.get('subreddit', '')}"

    post_text = f"Title: {title}\n\n{selftext}".strip()

    llm_result = call_llm_for_gap_analysis(post_text, comments_text)
    gap_dict = fill_gap_defaults(llm_result)
    rewrite_priority, optimization_goal = derive_rewrite_and_goal(row)

    return {
        'post_id': post_id,
        'source_keyword': row.get('source_keyword', ''),
        'subreddit': row.get('subreddit', ''),
        'primary_topic': row.get('primary_topic', ''),
        'content_angle': row.get('content_angle', ''),
        **gap_dict,
        'rewrite_priority': rewrite_priority,
        'optimization_goal': optimization_goal
    }


def run():
    logging.info('加载 post_comment_aggregate 和原始帖子特征...')
    df_posts = pd.read_csv(POST_COMMENT_AGG_PATH)
    df_comments = load_all_comment_labels(COMMENT_LABELS_DIR)
    df_post_features = load_all_post_features(POST_FEATURES_DIR)

    candidates = df_posts[(df_posts['question_ratio'] > MIN_QUESTION_RATIO) | (df_posts['correction_ratio'] > MIN_CORRECTION_RATIO)]
    if candidates.empty:
        logging.info('未找到候选帖子，退出。')
        return

    records = []
    max_workers = min(8, len(candidates))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_post = {
            executor.submit(process_candidate, row, df_comments, df_post_features): row['post_id']
            for _, row in candidates.iterrows()
        }
        for future in as_completed(future_to_post):
            post_id = future_to_post[future]
            try:
                record = future.result()
                records.append(record)
            except Exception as e:
                logging.error(f"处理 post_id={post_id} 时出错：{e}")

    df_out = pd.DataFrame(records)
    CONTENT_GAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(CONTENT_GAP_PATH, index=False, encoding='utf-8-sig')
    logging.info(f'生成 content_gap.csv 完成，路径：{CONTENT_GAP_PATH}，行数：{len(df_out)}')


if __name__ == '__main__':
    run()
