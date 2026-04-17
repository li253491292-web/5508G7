import os
import json
import logging
import time
import re
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import hashlib
from .llm_config import resolve_llm_config

# ---------- 配置 ----------
LLM_PROVIDER, API_KEY, LLM_URL, LLM_MODEL = resolve_llm_config()

# 路径配置 - 直接明确每个input和output的地址，方便后期直接修改
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = PROJECT_ROOT / "5506" / "cleaned"
POST_COMMENT_AGG_PATH = CLEANED_DIR / "aggregation" / "post_comment_aggregate.csv"
POST_FEATURES_DIR = CLEANED_DIR / "features"
COMMENT_LABELS_DIR = CLEANED_DIR / "labels"
CONTENT_GAP_PATH = CLEANED_DIR / "content_gap" / "content_gap.csv"

# 节省逻辑参数 - 大规模处理配置
MIN_COMMENTS = 5          # 至少有5条评论才值得分析Gap
MIN_QUESTION_RATIO = 0.3  # 问题比例阈值
MIN_CORRECTION_RATIO = 0.2 # 纠错比例阈值
LLM_LIMIT = 500           # 🚀 提升至500个高价值帖子进行LLM深度分析

# 大规模处理优化参数
BATCH_SIZE = 50           # 每批处理的帖子数量
MAX_WORKERS = 16          # 最大并发线程数（根据CPU核心数调整）
REQUEST_TIMEOUT = 45      # LLM请求超时时间（秒）
RETRY_ATTEMPTS = 3        # 失败重试次数
RATE_LIMIT_DELAY = 0.5    # 请求间隔延迟（秒）

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- 基础加载函数（带缓存优化） ----------
@lru_cache(maxsize=32)
def load_all_csv(directory: str, prefix: str) -> pd.DataFrame:
    """带缓存的CSV加载函数，避免重复I/O操作"""
    files = [os.path.join(directory, f) for f in os.listdir(directory) 
             if f.startswith(prefix) and f.endswith(".csv")]
    if not files: 
        return pd.DataFrame()
    
    # 使用低内存模式读取大文件
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            logging.warning(f"加载文件 {f} 失败: {e}")
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# 创建索引缓存避免重复搜索
_post_index_cache = {}

def get_post_context(df_posts: pd.DataFrame, post_id: str) -> Tuple[str, str]:
    """优化的帖子上下文获取，使用索引缓存"""
    cache_key = id(df_posts)
    
    # 构建或获取索引缓存
    if cache_key not in _post_index_cache:
        # 创建post_id到索引的映射
        _post_index_cache[cache_key] = {
            str(pid): idx for idx, pid in enumerate(df_posts['post_id'].astype(str))
        }
    
    index_map = _post_index_cache[cache_key]
    post_id_str = str(post_id)
    
    if post_id_str in index_map:
        idx = index_map[post_id_str]
        row = df_posts.iloc[idx]
        return str(row.get('title', '')), str(row.get('selftext', ''))
    
    return "Unknown Title", "No content available"

# 评论缓存
_comment_cache = {}

def get_top_comments(df_comments: pd.DataFrame, post_id: str, n: int = 5) -> str:
    """优化的评论获取，使用排序缓存"""
    cache_key = f"{id(df_comments)}_{post_id}_{n}"
    
    if cache_key in _comment_cache:
        return _comment_cache[cache_key]
    
    # 使用向量化操作而非逐行比较
    mask = df_comments['post_id'].astype(str) == str(post_id)
    subset = df_comments[mask]
    
    if subset.empty:
        result = ""
    else:
        # 使用nlargest替代sort_values提高性能
        top_c = subset.nlargest(n, 'comment_score')
        result = "\n---\n".join(top_c['comment_body'].astype(str).tolist())
    
    _comment_cache[cache_key] = result
    return result

# LLM调用缓存（扩展版）
_llm_cache = {}
_llm_stats = {'total_requests': 0, 'cache_hits': 0, 'failures': 0}

def _generate_cache_key(post_text: str, comments_text: str) -> str:
    """生成LLM调用缓存键"""
    content = f"{post_text}_{comments_text}"
    return hashlib.md5(content.encode()).hexdigest()

def call_llm_analysis(post_text: str, comments_text: str, post_id: str = None) -> Dict:
    """🚀 增强版LLM分析调用，支持大规模处理和智能重试"""
    cache_key = _generate_cache_key(post_text, comments_text)
    _llm_stats['total_requests'] += 1
    
    # 检查缓存
    if cache_key in _llm_cache:
        _llm_stats['cache_hits'] += 1
        logging.debug(f"缓存命中: {post_id}")
        return _llm_cache[cache_key]
    
    # 智能提示词优化 - 根据内容长度调整
    post_summary = post_text[:800] if len(post_text) > 800 else post_text
    comments_summary = comments_text[:800] if len(comments_text) > 800 else comments_text
    
    prompt = f"""作为医疗营销专家，分析以下医疗帖子内容与评论区的关键差异和优化机会。

📋 帖子内容：
{post_summary}

💬 代表性评论：
{comments_summary}

请提供结构化的内容缺口分析，包括：
1. 主要问题类型（安全性、效果、费用等）
2. 用户异议点
3. 缺失的关键信息
4. 信任信号缺口
5. 内容清晰度问题
6. 建议的优化策略

返回JSON格式，字段：dominant_question, dominant_objection, missing_info_types, missing_trust_signals, clarity_gap_summary, trust_gap_summary, recommended_tone, suggested_content_fix"""
    
    headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
    payload = {
        'model': LLM_MODEL,
        'messages': [{'role': 'system', 'content': prompt}],
        'response_format': {'type': 'json_object'},
        'temperature': 0.2,  # 降低随机性提高一致性
        'max_tokens': 1000   # 限制响应长度
    }
    
    # 智能重试机制
    for attempt in range(RETRY_ATTEMPTS):
        try:
            # 速率限制控制
            if attempt > 0:
                delay = RATE_LIMIT_DELAY * (2 ** attempt)  # 指数退避
                logging.info(f"等待 {delay:.1f} 秒后重试 (尝试 {attempt + 1}/{RETRY_ATTEMPTS})")
                time.sleep(delay)
            
            # 使用会话连接池提高性能
            with requests.Session() as session:
                r = session.post(LLM_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
                
                if r.status_code == 200:
                    response_data = r.json()
                    result = json.loads(response_data['choices'][0]['message']['content'])
                    
                    # 验证结果完整性
                    required_fields = ['dominant_question', 'dominant_objection', 'missing_info_types', 
                                     'missing_trust_signals', 'clarity_gap_summary', 'trust_gap_summary', 
                                     'recommended_tone', 'suggested_content_fix']
                    
                    if all(field in result for field in required_fields):
                        _llm_cache[cache_key] = result
                        return result
                    else:
                        logging.warning(f"LLM响应缺少必要字段: {post_id}")
                        
                elif r.status_code == 429:  # 速率限制
                    logging.warning(f"API速率限制触发，等待后重试: {post_id}")
                    continue
                else:
                    logging.error(f"LLM API错误 (状态码 {r.status_code}): {post_id}")
                    
        except requests.exceptions.Timeout:
            logging.error(f"LLM请求超时 (尝试 {attempt + 1}/{RETRY_ATTEMPTS}): {post_id}")
        except requests.exceptions.ConnectionError:
            logging.error(f"LLM连接错误 (尝试 {attempt + 1}/{RETRY_ATTEMPTS}): {post_id}")
        except json.JSONDecodeError as e:
            logging.error(f"JSON解析错误: {e} - {post_id}")
        except Exception as e:
            logging.error(f"LLM分析意外错误: {e} - {post_id}")
    
    _llm_stats['failures'] += 1
    logging.error(f"LLM分析失败 (所有重试用尽): {post_id}")
    
    # 返回保底结果
    return {
        'dominant_question': 'General information needs',
        'dominant_objection': 'Lack of detailed information',
        'missing_info_types': 'Comprehensive details and evidence',
        'missing_trust_signals': 'Authoritative sources and testimonials',
        'clarity_gap_summary': 'Unable to analyze due to API issues',
        'trust_gap_summary': 'Analysis failed - manual review needed',
        'recommended_tone': 'Professional and informative',
        'suggested_content_fix': 'Add detailed information and authoritative sources'
    }

# ---------- 智能启发式生成（数据驱动优化） ----------
def generate_heuristic_gap(row):
    """智能启发式生成 - 结合主题映射和数据驱动"""
    
    # 获取主题并标准化处理
    topic = str(row.get('dominant_comment_topic', 'General')).strip()
    if topic == 'nan' or not topic:
        topic = 'General'
    
    # 获取关键数据指标
    question_ratio = row.get('question_ratio', 0)
    correction_ratio = row.get('correction_ratio', 0)
    risk_ratio = row.get('risk_concern_ratio', 0)
    cost_ratio = row.get('cost_concern_ratio', 0)
    
    # 智能主题映射 - 基于数据指标动态调整
    topic_map = {
        'Safety': {
            'objection': ("Side effects & long-term risks", "Lack of safety protocols"),
            'priority': 'High' if risk_ratio > 0.2 else 'Medium',
            'focus': f"safety myths and {risk_ratio:.0%} risk concerns"
        },
        'Efficacy': {
            'objection': ("Mechanism validation & success rates", "Missing clinical evidence"),
            'priority': 'High' if correction_ratio > 0.15 else 'Medium',
            'focus': f"efficacy questions and {correction_ratio:.0%} scientific doubts"
        },
        'Cost': {
            'objection': ("Insurance coverage & hidden fees", "Cost-benefit transparency"),
            'priority': 'High' if cost_ratio > 0.25 else 'Medium',
            'focus': f"cost transparency and {cost_ratio:.0%} pricing concerns"
        },
        'Access': {
            'objection': ("Provider locations & wait times", "Logistical barriers"),
            'priority': 'Medium',
            'focus': "accessibility and practical information"
        }
    }
    
    # 获取主题配置（默认处理）
    config = topic_map.get(topic, {
        'objection': ("General procedural details", "Information specificity"),
        'priority': 'Low',
        'focus': f"{question_ratio:.0%} general questions and information gaps"
    })
    
    q, g = config['objection']
    priority = config['priority']
    focus_area = config['focus']
    
    # 数据驱动的智能内容生成
    if question_ratio > 0.4:
        question_type = "comprehensive details"
        content_strategy = "extensive FAQ section"
    elif question_ratio > 0.2:
        question_type = "specific clarifications"
        content_strategy = "targeted information blocks"
    else:
        question_type = "basic information"
        content_strategy = "concise summary section"
    
    return {
        'dominant_question': f"{topic}-specific {question_type}" if topic != 'General' else f"General {question_type}",
        'dominant_objection': q,
        'missing_info_types': f"{topic.lower()} protocols, verified case studies, practitioner credentials",
        'missing_trust_signals': f"Institutional endorsements, peer testimonials, {topic.lower()} success metrics",
        'clarity_gap_summary': f"{priority} priority: {focus_area} require detailed explanation (based on {question_ratio:.0%} question ratio).",
        'trust_gap_summary': g,
        'recommended_tone': "Empathetic & Evidence-based" if topic in ['Safety', 'Efficacy'] else "Informative & Accessible",
        'suggested_content_fix': f"Add {content_strategy} addressing {focus_area}, include verifiable sources and real case examples."
    }

# ---------- 主逻辑（性能优化版） ----------
def run_content_gap_process():
    """🚀 主流程：支持500个帖子的大规模内容缺口分析"""
    
    logging.info("🚀 开始生成 Content Gap (大规模处理版 - 500个帖子)...")
    overall_start_time = time.time()
    
    # 性能监控初始化
    performance_stats = {
        'start_time': overall_start_time,
        'data_load_time': 0,
        'llm_analysis_time': 0,
        'heuristic_time': 0,
        'save_time': 0,
        'total_processed': 0,
        'total_failed': 0
    }
    
    # 1. 加载数据（批量预加载）
    logging.info("📊 批量加载数据...")
    df_agg = pd.read_csv(POST_COMMENT_AGG_PATH)
    df_comments = load_all_csv(COMMENT_LABELS_DIR, "comment_feature_")  # 修正前缀
    df_post_raw = load_all_csv(POST_FEATURES_DIR, "post_feature_")
    
    # 数据预处理和验证 - 检查实际存在的列名
    df_agg = df_agg.dropna(subset=['post_id'])
    
    # 验证评论数据框的列名
    if not df_comments.empty:
        logging.info(f"评论数据列名: {list(df_comments.columns)}")
        # 只有在列存在时才进行dropna操作
        required_comment_cols = ['post_id', 'comment_body']
        available_comment_cols = [col for col in required_comment_cols if col in df_comments.columns]
        if available_comment_cols:
            df_comments = df_comments.dropna(subset=available_comment_cols)
        else:
            logging.warning(f"评论数据中未找到所需列: {required_comment_cols}")
    
    df_post_raw = df_post_raw.dropna(subset=['post_id']) if 'post_id' in df_post_raw.columns else df_post_raw
    
    logging.info(f"数据加载完成 - 帖子: {len(df_agg)}, 评论: {len(df_comments)}, 原始帖子: {len(df_post_raw)}")

    # 2. 筛选候选者（向量化优化）
    candidates = df_agg[
        (df_agg['total_comments'] >= MIN_COMMENTS) & 
        ((df_agg['question_ratio'] > MIN_QUESTION_RATIO) | (df_agg['correction_ratio'] > MIN_CORRECTION_RATIO))
    ].copy()
    
    logging.info(f"筛选后候选帖子: {len(candidates)} 个")

    if candidates.empty:
        logging.info("❌ 没有符合条件的候选帖子。")
        return

    # 3. 排序并分组（使用更高效的排序算法）
    candidates = candidates.sort_values(
        by=['controversy_score', 'total_comments'], 
        ascending=False,
        kind='mergesort'  # 稳定排序，大数据集性能更好
    )
    llm_tasks = candidates.head(LLM_LIMIT)
    auto_tasks = candidates.iloc[LLM_LIMIT:]
    
    logging.info(f"LLM分析组: {len(llm_tasks)} 个, 启发式组: {len(auto_tasks)} 个")

    records = []

    # 4. 🚀 大规模LLM深度分析（支持500个帖子）
    logging.info(f"🧠 正在对前 {len(llm_tasks)} 个核心帖子进行 LLM 深度分析...")
    
    # 预计算所有需要的数据，避免重复计算
    llm_data = []
    for idx, (_, row) in enumerate(llm_tasks.iterrows(), 1):
        p_title, p_body = get_post_context(df_post_raw, row['post_id'])
        c_text = get_top_comments(df_comments, row['post_id'])
        llm_data.append({
            'index': idx,
            'row': row,
            'post_text': f"{p_title}\n{p_body}",
            'comments_text': c_text,
            'post_id': row['post_id']
        })
    
    logging.info(f"📊 预计算完成，共 {len(llm_data)} 个分析任务")
    
    # 批量处理配置
    total_batches = (len(llm_data) + BATCH_SIZE - 1) // BATCH_SIZE
    processed_count = 0
    failed_count = 0
    
    # 使用优化的线程池配置
    optimal_workers = min(MAX_WORKERS, len(llm_data), (os.cpu_count() or 1) * 4)
    logging.info(f"⚙️ 使用 {optimal_workers} 个工作线程，批量大小 {BATCH_SIZE}")
    
    with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
        # 分批处理，避免内存溢出
        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(llm_data))
            batch_data = llm_data[start_idx:end_idx]
            
            logging.info(f"📦 处理第 {batch_idx + 1}/{total_batches} 批，包含 {len(batch_data)} 个任务")
            
            # 提交当前批次任务
            batch_futures = []
            for data in batch_data:
                future = executor.submit(
                    call_llm_analysis, 
                    data['post_text'], 
                    data['comments_text'],
                    data['post_id']
                )
                batch_futures.append((future, data['row'], data['index']))
            
            # 收集当前批次结果
            for future, row, index in batch_futures:
                try:
                    res = future.result(timeout=REQUEST_TIMEOUT)
                    if res and res.get('dominant_question') != 'General information needs':  # 确保有有效结果
                        res.update({
                            'post_id': row['post_id'], 
                            'rewrite_priority': 'High',
                            'optimization_goal': 'Deep Strategy',
                            'analysis_batch': batch_idx + 1
                        })
                        records.append(res)
                        processed_count += 1
                        
                        # 每50个显示进度
                        if processed_count % 50 == 0:
                            logging.info(f"✅ 已完成 {processed_count}/{len(llm_tasks)} 个LLM分析")
                    else:
                        failed_count += 1
                        logging.warning(f"⚠️ 无效分析结果: {row['post_id']} (索引 {index})")
                        
                except Exception as e:
                    failed_count += 1
                    logging.error(f"❌ LLM分析失败: {row['post_id']} (索引 {index}): {e}")
            
            # 批次间短暂休息，避免API过载
            if batch_idx < total_batches - 1:
                logging.info(f"⏱️ 批次间休息 {RATE_LIMIT_DELAY} 秒...")
                time.sleep(RATE_LIMIT_DELAY)
    
    # LLM分析统计
    success_rate = processed_count / len(llm_tasks) * 100 if len(llm_tasks) > 0 else 0
    logging.info(f"📈 LLM分析完成统计:")
    logging.info(f"   ✅ 成功: {processed_count} 个")
    logging.info(f"   ❌ 失败: {failed_count} 个")
    logging.info(f"   📊 成功率: {success_rate:.1f}%")
    logging.info(f"   💾 缓存命中率: {_llm_stats['cache_hits']}/{_llm_stats['total_requests']} ({_llm_stats['cache_hits']/_llm_stats['total_requests']*100:.1f}%)")

    # 5. 执行启发式自动填充 (0 成本)
    logging.info(f"🤖 正在对剩余 {len(auto_tasks)} 个帖子执行启发式自动分析...")
    for _, row in auto_tasks.iterrows():
        h_res = generate_heuristic_gap(row)
        h_res.update({
            'post_id': row['post_id'],
            'rewrite_priority': 'Medium' if row['total_comments'] > 10 else 'Low',
            'optimization_goal': 'Heuristic Fix'
        })
        records.append(h_res)

    # 6. 📊 最终统计和保存（增强版）
    save_start_time = time.time()
    
    if records:
        final_df = pd.DataFrame(records)
        
        # 使用更高效的数据合并 - 修正列名错误
        available_cols = ['post_id', 'source_keyword', 'subreddit']
        if 'dominant_comment_topic' in df_agg.columns:
            available_cols.append('dominant_comment_topic')
        elif 'primary_topic' in df_agg.columns:
            available_cols.append('primary_topic')
        else:
            logging.warning("未找到主题相关列，将使用可用列进行合并")
        
        base_info = df_agg[available_cols].copy()
        final_df = pd.merge(final_df, base_info, on='post_id', how='left')
        
        # 确保目录存在并批量写入
        CONTENT_GAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(CONTENT_GAP_PATH, index=False, encoding='utf-8-sig')
        
        performance_stats['save_time'] = time.time() - save_start_time
        performance_stats['total_processed'] = len(records)
        
        # 📈 详细性能报告
        total_time = time.time() - overall_start_time
        
        logging.info("\n" + "="*60)
        logging.info("🎯 大规模内容缺口分析完成报告")
        logging.info("="*60)
        logging.info(f"📊 总体统计:")
        logging.info(f"   ✅ 成功处理: {len(records)} 个帖子")
        logging.info(f"   ⚡ 总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
        logging.info(f"   📈 平均速度: {len(records)/total_time:.2f} 帖子/秒")
        logging.info(f"   💾 输出文件: {CONTENT_GAP_PATH}")
        
        logging.info(f"\n⏱️ 时间分布:")
        logging.info(f"   📋 数据加载: {performance_stats['data_load_time']:.1f}s")
        logging.info(f"   🤖 LLM分析: {performance_stats['llm_analysis_time']:.1f}s")
        logging.info(f"   ⚙️ 启发式分析: {performance_stats['heuristic_time']:.1f}s")
        logging.info(f"   💾 文件保存: {performance_stats['save_time']:.1f}s")
        
        # LLM分析详细统计
        if _llm_stats['total_requests'] > 0:
            logging.info(f"\n🤖 LLM分析统计:")
            logging.info(f"   📞 总请求数: {_llm_stats['total_requests']}")
            logging.info(f"   💾 缓存命中: {_llm_stats['cache_hits']} ({_llm_stats['cache_hits']/_llm_stats['total_requests']*100:.1f}%)")
            logging.info(f"   ❌ 失败数: {_llm_stats['failures']}")
            logging.info(f"   ✅ 成功率: {(_llm_stats['total_requests']-_llm_stats['failures'])/_llm_stats['total_requests']*100:.1f}%")
        
        # 数据质量分析
        high_priority = len(final_df[final_df['rewrite_priority'] == 'High'])
        medium_priority = len(final_df[final_df['rewrite_priority'] == 'Medium'])
        
        logging.info(f"\n🎯 优先级分布:")
        logging.info(f"   🔴 高优先级: {high_priority} 个 ({high_priority/len(final_df)*100:.1f}%)")
        logging.info(f"   🟡 中优先级: {medium_priority} 个 ({medium_priority/len(final_df)*100:.1f}%)")
        logging.info(f"   🟢 低优先级: {len(final_df)-high_priority-medium_priority} 个")
        
        logging.info("="*60)
        
    else:
        performance_stats['total_failed'] = 1
        logging.warning("⚠️ 没有生成任何分析结果 - 请检查输入数据和配置参数")

if __name__ == "__main__":
    run_content_gap_process()
