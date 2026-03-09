# 演示 RAG（检索增强生成）的轻量级实现方案
# 使用 BM25 关键词检索（无需向量数据库，无需 Embedding 模型，适合快速上手）
#
# BM25 vs ChromaDB（rag_using_chromadb.py）对比：
#   BM25：基于关键词频率的词法检索，速度快，无需 GPU，但无法理解语义
#   ChromaDB：基于向量相似度的语义检索，效果更好，但需要 Embedding 模型
#
# 前置依赖：pip install smolagents langchain langchain-community rank_bm25 datasets

# from huggingface_hub import login
# login()  # 如果数据集是私有的，需要先登录 HuggingFace

import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever  # 基于 BM25 算法的关键词检索器


# ============================================================
# 第一步：加载并过滤知识库文档
# ============================================================

# 从 HuggingFace Hub 加载 HuggingFace 官方文档数据集
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

# 只保留 transformers 库相关的文档（缩小知识库范围，加快检索速度）
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

# 转换为 LangChain Document 格式
# metadata["source"] 记录文档来源，便于溯源
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]}) for doc in knowledge_base
]


# ============================================================
# 第二步：文档分块（Chunking）
# ============================================================

# RecursiveCharacterTextSplitter：按字符数分块（而非 token 数）
# 相比 rag_using_chromadb.py 中按 token 分块的方式更简单，但精度略低
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每个 chunk 最多 500 个字符
    chunk_overlap=50,      # 相邻 chunk 重叠 50 个字符，避免在边界处截断语义
    add_start_index=True,  # 在 metadata 中记录每个 chunk 在原文中的起始位置
    strip_whitespace=True, # 去除首尾空白
    separators=["\n\n", "\n", ".", " ", ""],  # 优先在段落/句子边界处分割
)
docs_processed = text_splitter.split_documents(source_docs)


# ============================================================
# 第三步：将 BM25 检索器封装为 smolagents Tool
# ============================================================

from smolagents import Tool


class RetrieverTool(Tool):
    """将 BM25 关键词检索能力封装为 Agent 可调用的工具"""
    name = "retriever"
    description = "Uses lexical search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            # 提示 LLM 用陈述句（而非疑问句）查询，更贴近文档的表达方式
            # 例如："transformers training backward pass" 比 "which is slower?" 效果更好
            "description": "The query to perform. This should be lexically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        # BM25Retriever 会在初始化时对所有文档建立索引（内存中，无需持久化）
        # k=10：每次检索返回最相关的 10 个文档块
        self.retriever = BM25Retriever.from_documents(docs, k=10)

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        # 执行 BM25 检索，返回最相关的文档块列表
        docs = self.retriever.invoke(query)

        # 将检索到的文档块格式化为字符串，方便 LLM 阅读和引用
        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )


# ============================================================
# 第四步：创建 Agent 并运行
# ============================================================

from smolagents import CodeAgent, InferenceClientModel


# 实例化检索工具（传入分块后的文档列表）
retriever_tool = RetrieverTool(docs_processed)

# 创建 CodeAgent
# max_steps=4：RAG 场景通常 1-2 步即可完成，4 步已足够
# verbosity_level=2：打印详细日志，可以看到 Agent 的检索查询和推理过程
agent = CodeAgent(
    tools=[retriever_tool],
    model=InferenceClientModel(model_id="Qwen/Qwen3-Next-80B-A3B-Thinking"),
    max_steps=4,
    verbosity_level=2,
    stream_outputs=True,
)

# 运行 Agent：它会构造合适的查询词，调用 retriever 工具检索相关文档，
# 再基于检索结果回答问题（而非依赖 LLM 的训练知识）
agent_output = agent.run("For a transformers model training, which is slower, the forward or the backward pass?")

print("Final output:")
print(agent_output)
