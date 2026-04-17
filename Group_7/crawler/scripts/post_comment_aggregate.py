import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from pathlib import Path

# ================= 配置区 =================
# 删除了 LLM 相关配置，因为本步骤是纯本地逻辑聚合，无需联网
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = PROJECT_ROOT / "5506" / "cleaned"
POST_FEATURES_DIR = CLEANED_DIR / "features"
COMMENT_LABELS_DIR = CLEANED_DIR / "labels"
OUTPUT_PATH = CLEANED_DIR / "aggregation"

MIN_COMMENTS_THRESHOLD = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_all_files(directory, prefix):
    """通用加载器：加载目录下所有以指定前缀开头的CSV"""
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".csv")]
    if not all_files:
        logging.warning(f"⚠️ 在 {directory} 未找到前缀为 {prefix} 的文件")
        return pd.DataFrame()
    
    logging.info(f"正在加载 {len(all_files)} 个 {prefix} 相关文件...")
    
    dfs = []
    for file_path in all_files:
        try:
            # Try reading with error handling for malformed lines
            df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8')
            dfs.append(df)
            logging.info(f"✅ 成功加载: {os.path.basename(file_path)} (行数: {len(df)})")
        except Exception as e:
            logging.error(f"❌ 加载失败: {os.path.basename(file_path)} - {str(e)}")
            continue
    
    if not dfs:
        logging.error("所有文件加载失败")
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)

def calculate_aggregate():
    # 1. 加载数据
    df_posts = load_all_files(POST_FEATURES_DIR, "post_feature_")
    df_comments = load_all_files(COMMENT_LABELS_DIR, "comment_feature_")
    
    if df_posts.empty or df_comments.empty:
        logging.error("数据加载失败，请检查路径。")
        return

    # 2. 评论预处理：统一 ID 类型和处理缺失
    df_comments['post_id'] = df_comments['post_id'].astype(str)
    df_posts['post_id'] = df_posts['post_id'].astype(str)

    # 3. 基础布尔字段聚合 (均值即比例)
    logging.info("执行数值维度统计...")
    agg_logic = {
        'comment_id': 'count',
        'is_top_level': 'sum',
        'depth': ['max', 'mean'],
        'comment_score': ['mean', 'max', 'min'],
        'contains_question': 'mean',
        'contains_advice': 'mean',
        'contains_personal_experience': 'mean',
        'contains_correction': 'mean',
        'contains_recommendation': 'mean',
        'contains_risk_concern': 'mean',
        'contains_cost_concern': 'mean',
        'contains_efficacy_concern': 'mean',
        'emotional_intensity': 'mean'
    }
    
    comment_grouped = df_comments.groupby('post_id').agg(agg_logic)
    comment_grouped.columns = [
        'total_comments', 'top_level_comment_count', 'max_depth', 'avg_depth',
        'avg_comment_score', 'max_comment_score', 'min_comment_score',
        'question_ratio', 'advice_ratio', 'personal_experience_ratio',
        'correction_ratio', 'recommendation_ratio', 'risk_concern_ratio',
        'cost_concern_ratio', 'efficacy_concern_ratio', 'avg_emotional_intensity'
    ]
    comment_grouped['reply_comment_count'] = comment_grouped['total_comments'] - comment_grouped['top_level_comment_count']

    # 4. 提取分类变量的众数 (Dominant Topic/Emotion)
    def get_mode_safe(x):
        clean_x = x.dropna().astype(str).str.strip()
        clean_x = clean_x[~clean_x.isin(['Unknown', 'None', 'nan', ''])]
        return clean_x.mode().iloc[0] if not clean_x.empty else "Unknown"

    mode_features = df_comments.groupby('post_id').agg({
        'topic': get_mode_safe,
        'emotion': get_mode_safe
    }).rename(columns={'topic': 'dominant_comment_topic', 'emotion': 'dominant_comment_emotion'})

    # 5. 立场与信任信号比例计算 (优化映射逻辑)
    def get_ratio_mapping(df, group_col, target_col, mapping_dict):
        # 统一大小写处理，增强容错
        temp_df = df.copy()
        temp_df[target_col] = temp_df[target_col].astype(str).str.capitalize()
        counts = temp_df.groupby([group_col, target_col]).size().unstack(fill_value=0)
        ratios = counts.div(counts.sum(axis=1), axis=0)
        # 只保留 mapping 中定义的列并重命名
        existing_cols = [c for c in mapping_dict.keys() if c in ratios.columns]
        return ratios[existing_cols].rename(columns=mapping_dict)

    stance_map = {'Supportive': 'support_ratio', 'Skeptical': 'skeptical_ratio', 'Opposed': 'oppose_ratio', 'Neutral': 'neutral_ratio'}
    trust_map = {
        'Doctor_background': 'trust_signal_doctor_ratio', 
        'Scientific_evidence': 'trust_signal_evidence_ratio',
        'Peer_anecdotes': 'trust_signal_anecdote_ratio', 
        'Institution_rank': 'trust_signal_institution_ratio'
    }

    stance_ratios = get_ratio_mapping(df_comments, 'post_id', 'stance', stance_map)
    trust_ratios = get_ratio_mapping(df_comments, 'post_id', 'trust_signal', trust_map)

    # 6. 合并所有评论统计项
    comment_final_stats = pd.concat([comment_grouped, mode_features, stance_ratios, trust_ratios], axis=1).fillna(0)

    # 7. 关联主帖
    final_df = pd.merge(df_posts, comment_final_stats, left_on='post_id', right_index=True, how='inner')
    final_df = final_df[final_df['total_comments'] >= MIN_COMMENTS_THRESHOLD]

    # 8. 衍生洞察指标计算 (核心逻辑对撞)
    logging.info("基于标签对撞计算衍生指标...")
    
    # 争议得分：(Skeptical + Oppose) vs Support 的平衡度
    # 若怀疑和支持各占一半，得分趋近1
    final_df['controversy_score'] = 1 - abs(final_df.get('support_ratio', 0) - (final_df.get('skeptical_ratio', 0) + final_df.get('oppose_ratio', 0)))
    
    # 讨论强度：评论量 * 互动深度
    final_df['discussion_intensity_score'] = np.log1p(final_df['total_comments']) * (final_df['max_depth'] + 1)

    # 信任鸿沟 (Trust Gap)：
    # 逻辑：如果帖子本身信息缺失严重(gap_level高) 且 评论区出现了大量 纠错(correction) 或 风险担忧(risk)
    final_df['trust_gap_score'] = final_df['information_gap_level'] * (final_df['correction_ratio'] + final_df['risk_concern_ratio'])

    # 清晰度鸿沟 (Clarity Gap)：
    # 逻辑：发帖质量低 且 评论区全是问题(question)
    final_df['clarity_gap_score'] = final_df['information_gap_level'] * final_df['question_ratio']

    # 9. 最终列对齐与保存
    required_cols = [
        'post_id', 'source_keyword', 'subreddit', 'total_comments', 'top_level_comment_count', 
        'reply_comment_count', 'max_depth', 'avg_depth', 'avg_comment_score', 'max_comment_score', 
        'min_comment_score', 'support_ratio', 'skeptical_ratio', 'oppose_ratio', 'neutral_ratio',
        'question_ratio', 'advice_ratio', 'personal_experience_ratio', 'correction_ratio', 
        'recommendation_ratio', 'risk_concern_ratio', 'cost_concern_ratio', 'efficacy_concern_ratio',
        'trust_signal_doctor_ratio', 'trust_signal_evidence_ratio', 'trust_signal_anecdote_ratio', 
        'trust_signal_institution_ratio', 'controversy_score', 'discussion_intensity_score', 
        'trust_gap_score', 'clarity_gap_score', 'dominant_comment_topic', 'dominant_comment_emotion'
    ]

    # 补齐缺失列并导出
    for col in required_cols:
        if col not in final_df.columns: final_df[col] = 0.0 if '_ratio' in col or '_score' in col else "Unknown"
    
    output_file = OUTPUT_PATH / "post_comment_aggregate.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df[required_cols].to_csv(output_file, index=False, encoding='utf-8-sig')
    
    logging.info(f"✅ 聚合完成！最终有效帖子数: {len(final_df)}。")

if __name__ == "__main__":
    calculate_aggregate()
