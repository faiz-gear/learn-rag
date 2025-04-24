from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.callbacks import CallbackManager
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

langfuse_callback_handler = LlamaIndexCallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_API_HOST"),
)

callback_manager = CallbackManager([langfuse_callback_handler])
Settings.callback_manager = callback_manager

from llama_index.core import PropertyGraphIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core.graph_stores import SimplePropertyGraphStore

documents = SimpleDirectoryReader(
    input_files=[os.path.join(os.path.dirname(__file__), "../../data/graph.txt")]
).load_data()

# 指定知识图谱的存储，这里使用内存存储，实际使用商业的图数据库（Neo4j、TigerGraph）
property_graph_store = SimplePropertyGraphStore()
storage_context = StorageContext.from_defaults(
    property_graph_store=property_graph_store
)

# 构造图谱索引
if not os.path.exists(f"./storage/graph_store"):
    index = PropertyGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    index.storage_context.persist(persist_dir="./storage/graph_store")
else:
    print("load from existing graph store")
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./storage/graph_store")
    )

# 使用图谱索引进行查询
query_engine = index.as_query_engine(include_text=True, similarity_top_k=2)
response = query_engine.query("介绍一下苹果公司")
print(response)
