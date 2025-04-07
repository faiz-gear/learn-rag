import chromadb

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")

# 设置模型
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 设置回调
llm_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llm_debug])
Settings.callback_manager = callback_manager

# 加载与读取文档
# 处理路径问题
relative_path = os.path.join(os.path.dirname(__file__), "../data/mcp.txt")
reader = SimpleDirectoryReader(input_files=[relative_path])
documents = reader.load_data()

# 分割文档
node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)

# 创建向量存储
chroma = chromadb.HttpClient(host="localhost", port=8000)

# try:
#     if chroma.get_collection(name="ragdb"):
#         chroma.delete_collection(name="ragdb")
# except Exception as e:
#     print(f"Error deleting collection: {e}")

collection = chroma.get_or_create_collection(
    name="ragdb", metadata={"hnsw:space": "cosine"}
)
chroma_vector_store = ChromaVectorStore(
    chroma_collection=collection,
)

# 准备向量存储索引
storage_context = StorageContext.from_defaults(vector_store=chroma_vector_store)
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
)

# 构造查询引擎
query_engine = index.as_query_engine()

while True:
    user_input = input("Enter a query: ")
    if user_input.lower() == "exit":
        break

    response = query_engine.query(user_input)

    print("Answer: ", response)
    print("Trace: ", llm_debug.print_trace_map())
