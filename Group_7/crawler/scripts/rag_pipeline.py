import pandas as pd
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore   # 新版推荐
import chromadb

# ================== 配置（已优化） ==================
# 使用 BGE-M3，多语言支持更好，适合你的 Reddit 数据
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

Settings.embed_model = embed_model
Settings.chunk_size = 512          # 可根据实际效果后续调整为 400~600
Settings.chunk_overlap = 80        # 稍微增加重叠，提升上下文连贯性

# 数据路径
POSTS_CSV = "post_final_travel_to_china_cleaned.csv"
COMMENTS_CSV = "comments_final_travel_to_china_cleaned.csv"

# ================== 步骤1：加载并合并数据 ==================
print("正在加载 travel_to_china 数据...")
posts_df = pd.read_csv(POSTS_CSV)
comments_df = pd.read_csv(COMMENTS_CSV)

documents = []
for _, post in posts_df.iterrows():
    post_text = f"Title: {post.get('title', '')}\n"
    post_text += f"Post Content: {post.get('selftext', '')}\n"
    
    # 附加该帖子的所有评论
    related_comments = comments_df[comments_df['post_id'] == post['post_id']]
    if not related_comments.empty:
        comments_str = "\n".join([f"Comment: {c.get('comment_body', '')}" for _, c in related_comments.iterrows()])
        post_text += f"\nComments:\n{comments_str}\n"
    
    doc = Document(
        text=post_text.strip(),   # 去掉多余空格
        metadata={
            "post_id": str(post.get('post_id')),
            "subreddit": post.get('subreddit'),
            "source_keyword": post.get('source_keyword'),
            "created_utc": str(post.get('created_utc')),
            "score": int(post.get('score', 0)),
            "num_comments": int(post.get('num_comments', 0)),
            "url": post.get('url', ''),
            "keyword_group": "travel_to_china",   # 重要：用于未来区分不同组
            "topics": "travel_experience"
        }
    )
    documents.append(doc)

print(f"✅ 总共加载 {len(documents)} 条文档（帖子 + 评论）")

# ================== 步骤2：创建专属向量数据库 ==================
print("正在初始化向量数据库...")
db_path = "./chroma_db_travel_to_china"   # ← 修改为专属路径（推荐修改）

db = chromadb.PersistentClient(path=db_path)
chroma_collection = db.get_or_create_collection("china_travel_rag")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ================== 步骤3：构建索引 ==================
print("正在构建索引（首次运行可能需要几分钟，请耐心等待）...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True
)

print(f"✅ 索引构建完成！向量数据库保存在：{db_path}")

# ================== 步骤4：测试查询引擎 ==================
print("\n正在创建查询引擎（使用本地模型）...")

# 使用本地 HuggingFace LLM（不需要 OpenAI Key）
# 使用本地 HuggingFace LLM（不需要 OpenAI Key）
from llama_index.llms.huggingface import HuggingFaceLLM

llm = HuggingFaceLLM(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",   # 更小、更快，适合首次测试
    device_map="auto",
    trust_remote_code=True,
    generate_kwargs={"temperature": 0.7, "max_new_tokens": 512},
    model_kwargs={"torch_dtype": "float16"},   # Mac 上更省内存
)

query_engine = index.as_query_engine(
    similarity_top_k=8,
    llm=llm,
    response_mode="compact"   # 更简洁稳定
)

print("\n=== 测试查询（基于 travel_to_china 数据） ===")
test_queries = [
    "提取海外用户在中国旅行中遇到的语言障碍具体例子和用户特征",
    "总结 solo 女性旅行者在中国遇到的安全、隐私或拥挤相关经历",
    "支付 App（Alipay / WeChat）、Didi 和 VPN 使用中的常见痛点有哪些？",
    "用户对中国高铁、机场、酒店等基础设施的真实评价是什么？",
]

for q in test_queries:
    print(f"\n问题: {q}")
    response = query_engine.query(q)
    print(f"回答: {response}")
    print("-" * 80)

print("\n🎉 RAG 系统已就绪！")