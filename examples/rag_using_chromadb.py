# 演示 RAG（检索增强生成）+ smolagents 的完整集成方案
# 使用 ChromaDB 作为向量数据库（语义搜索），比 rag.py 中的 BM25 关键词搜索更智能
#
# RAG 工作流程：
#   文档 → 分块(Chunking) → 向量化(Embedding) → 存入 ChromaDB
#   用户提问 → 语义检索相关文档块 → 将文档块注入 LLM 上下文 → 生成答案
#
# 前置依赖：
#   pip install smolagents langchain langchain-chroma langchain-huggingface
#             sentence-transformers datasets transformers tqdm

import os

import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma                           # ChromaDB 向量数据库集成

# from langchain_community.document_loaders import PyPDFLoader  # 如需加载 PDF 取消注释
from langchain_huggingface import HuggingFaceEmbeddings       # 本地 Embedding 模型
from tqdm import tqdm                                         # 进度条显示
from transformers import AutoTokenizer

# from langchain_openai import OpenAIEmbeddings  # 备选：使用 OpenAI Embedding
from smolagents import LiteLLMModel, Tool
from smolagents.agents import CodeAgent
# from smolagents.agents import ToolCallingAgent


# ============================================================
# 第一步：加载知识库文档
# ============================================================

# 从 HuggingFace Hub 加载预构建的文档数据集（HuggingFace 官方文档）
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

# 将数据集转换为 LangChain Document 格式，方便后续分块和向量化
# metadata 中保存文档来源，便于溯源
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]}) for doc in knowledge_base
]

## 如果你有自己的 PDF 文件，可以用以下代码加载（取消注释）：
# pdf_directory = "pdfs"
# pdf_files = [
#     os.path.join(pdf_directory, f)
#     for f in os.listdir(pdf_directory)
#     if f.endswith(".pdf")
# ]
# source_docs = []
# for file_path in pdf_files:
#     loader = PyPDFLoader(file_path)
#     docs.extend(loader.load())


# ============================================================
# 第二步：文档分块（Chunking）
# ============================================================

# 使用与 Embedding 模型配套的 Tokenizer 来精确控制 chunk 大小（按 token 数而非字符数）
# 这样可以避免 chunk 超出 Embedding 模型的最大输入长度
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained("thenlper/gte-small"),  # 与下方 Embedding 模型配套
    chunk_size=200,        # 每个 chunk 最多 200 个 token（较小，检索更精准）
    chunk_overlap=20,      # 相邻 chunk 重叠 20 个 token，避免在边界处截断语义
    add_start_index=True,  # 在 metadata 中记录每个 chunk 在原文中的起始位置
    strip_whitespace=True, # 去除首尾空白
    separators=["\n\n", "\n", ".", " ", ""],  # 优先在段落/句子边界处分割
)

# 分块并去重（相同内容的 chunk 只保留一份）
print("Splitting documents...")
docs_processed = []
unique_texts = {}
for doc in tqdm(source_docs):
    new_docs = text_splitter.split_documents([doc])
    for new_doc in new_docs:
        if new_doc.page_content not in unique_texts:
            unique_texts[new_doc.page_content] = True
            docs_processed.append(new_doc)


# ============================================================
# 第三步：向量化（Embedding）并存入 ChromaDB
# ============================================================

print("Embedding documents... This should take a few minutes (5 minutes on MacBook with M1 Pro)")

# 使用本地 Sentence Transformers 模型生成向量（无需 API Key，完全离线）
# all-MiniLM-L6-v2：轻量高效，384 维向量，适合中等规模知识库
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 备选：使用 OpenAI Embedding（效果更好但需要 API Key 和费用）
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 将所有 chunk 向量化后存入 ChromaDB
# persist_directory：持久化到本地磁盘，下次运行无需重新 Embedding
vector_store = Chroma.from_documents(docs_processed, embeddings, persist_directory="./chroma_db")


# ============================================================
# 第四步：将 ChromaDB 检索器封装为 smolagents Tool
# ============================================================

class RetrieverTool(Tool):
    """将向量数据库的语义检索能力封装为 Agent 可调用的工具"""
    name = "retriever"
    description = (
        "Uses semantic search to retrieve the parts of documentation that could be most relevant to answer your query."
    )
    inputs = {
        "query": {
            "type": "string",
            # 提示 LLM 用陈述句而非疑问句作为查询，更贴近文档的表达方式
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vector_store, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store  # 注入向量数据库实例

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        # k=3：返回最相关的 3 个文档块（可根据需要调整）
        docs = self.vector_store.similarity_search(query, k=3)
        # 将检索到的文档块格式化为字符串，方便 LLM 阅读
        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )


# 实例化检索工具
retriever_tool = RetrieverTool(vector_store)


# ============================================================
# 第五步：创建 Agent 并运行
# ============================================================

# 选择 LLM 后端（三种方案任选其一）：

# 方案1：HuggingFace 推理 API
# from smolagents import InferenceClientModel
# model = InferenceClientModel(model_id="Qwen/Qwen3-Next-80B-A3B-Thinking")

# 方案2：本地 Transformers 模型
# from smolagents import TransformersModel
# model = TransformersModel(model_id="Qwen/Qwen3-4B-Instruct-2507")

# 方案3（当前）：通过 LiteLLM 调用 Groq 托管的模型（需要 GROQ_API_KEY 环境变量）
# 切换 Anthropic：将 model_id 改为 'anthropic/claude-4-sonnet-latest'，api_key 改为 ANTHROPIC_API_KEY
model = LiteLLMModel(
    model_id="groq/openai/gpt-oss-120b",
    api_key=os.environ.get("GROQ_API_KEY"),
)

# 备选：ToolCallingAgent（JSON 格式调用工具）
# agent = ToolCallingAgent(
#     tools=[retriever_tool],
#     model=model,
#     verbose=True,
# )

# CodeAgent：LLM 生成代码来调用检索工具，支持多次检索和结果综合
# max_steps=4：RAG 场景通常 1-2 步即可完成，4 步已足够，避免过多 LLM 调用
agent = CodeAgent(
    tools=[retriever_tool],
    model=model,
    max_steps=4,
    verbosity_level=2,   # 打印详细日志，方便观察 Agent 的检索和推理过程
    stream_outputs=True,
)

# 运行 Agent：它会自动调用 retriever 工具检索相关文档，再基于文档内容回答问题
agent_output = agent.run("How can I push a model to the Hub?")


print("Final output:")
print(agent_output)
