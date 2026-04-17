import pandas as pd
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import chromadb
import os

# ================== 配置 ==================
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

Settings.embed_model = embed_model
Settings.chunk_size = 600
Settings.chunk_overlap = 80

# 数据路径
POSTS_CSV = "post_final_medical_tourism_cleaned.csv"
COMMENTS_CSV = "comments_final_medical_tourism_cleaned.csv"

# ================== 步骤1：加载并合并数据 ==================
print("正在加载 medical tourism 数据...")
posts_df = pd.read_csv(POSTS_CSV)
comments_df = pd.read_csv(COMMENTS_CSV)

documents = []
for _, post in posts_df.iterrows():
    post_text = f"Title: {post.get('title', '')}\n"
    post_text += f"Post Content: {post.get('selftext', '')}\n"

    # 附加该帖子的评论
    related_comments = comments_df[comments_df['post_id'] == post['post_id']]
    if not related_comments.empty:
        comments_str = "\n".join(
            [f"Comment: {c.get('comment_body', '')}" for _, c in related_comments.iterrows()])
        post_text += f"\nComments:\n{comments_str}\n"

    doc = Document(
        text=post_text,
        metadata={
            "post_id": str(post.get('post_id')),
            "subreddit": post.get('subreddit'),
            "source_keyword": post.get('source_keyword'),
            "created_utc": str(post.get('created_utc')),
            "score": int(post.get('score', 0)),
            "num_comments": int(post.get('num_comments', 0)),
            "url": post.get('url', ''),
            "topics": "medical_tourism",
            "procedure_type": str(post.get('selftext', ''))[:300]
        }
    )
    documents.append(doc)

print(f"总共加载 {len(documents)} 条文档（medical tourism 帖子 + 评论）")

# ================== 步骤2：创建向量数据库 ==================
print("正在初始化向量数据库...")
db_path = "./chroma_db_medical_tourism"
db = chromadb.PersistentClient(path=db_path)
chroma_collection = db.get_or_create_collection("medical_tourism_rag")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ================== 步骤3：构建索引 ==================
print("正在构建/加载索引（这可能需要几分钟）...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True
)

print("索引构建完成！向量数据库保存在 ./chroma_db_medical_tourism 文件夹")

# ================== 设置本地 LLM（已优化） ==================
print("正在加载本地 LLM 模型（首次运行会下载，请耐心等待）...")
llm = HuggingFaceLLM(
    model_name="Qwen/Qwen2.5-3B-Instruct",     # 推荐使用 3B 版本，回答质量更好
    context_window=8000,
    max_new_tokens=1024,
    generate_kwargs={"temperature": 0.7, "top_p": 0.95},
    tokenizer_name="Qwen/Qwen2.5-3B-Instruct",
)

Settings.llm = llm

# ================== 创建查询引擎 ==================
query_engine = index.as_query_engine(similarity_top_k=6)

# ================== 测试查询并保存结果 ==================
print("\n=== Medical Tourism 测试查询 ===")

test_queries = [
    "用户做医疗旅游最常去哪些国家？真实经历和评价如何？",
    "医疗旅游中常见的风险、并发症和后悔经历有哪些？请提取具体例子",
    "牙科植入（dental implants）、整容手术、关节置换等医疗旅游的费用对比和用户反馈",
    "在中国、土耳其、墨西哥、印度做医疗旅游的优缺点对比",
    "solo 患者或预算有限的人进行医疗旅游的痛点、注意事项和建议",
    "医疗旅游后回国后的跟进护理、并发症处理相关经历"
]

results = []

for q in test_queries:
    print(f"\n问题: {q}")
    response = query_engine.query(q)
    answer = str(response)

    print(f"回答: {answer}\n{'-'*80}")

    # 保存到列表
    results.append(f"问题: {q}\n回答: {answer}\n{'-'*80}\n")

# ================== 把回答保存到文件 ==================
output_file = "medical_tourism_rag_results.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("Medical Tourism RAG 查询结果\n")
    f.write("="*80 + "\n\n")
    f.write("\n".join(results))

print(f"\n查询结果已保存到文件: {output_file}")
print("Medical Tourism RAG 系统已就绪！")
