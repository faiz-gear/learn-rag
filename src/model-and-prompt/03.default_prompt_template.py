import chromadb

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import PromptTemplate
import pprint

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")

# 设置模型
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 加载与读取文档
# 处理路径问题
relative_path = os.path.join(os.path.dirname(__file__), "../../data/mcp.txt")
reader = SimpleDirectoryReader(input_files=[relative_path])
documents = reader.load_data()

# 分割文档
node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)

# 创建向量存储
chroma = chromadb.PersistentClient(path="./chroma_db")

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
prompts_dict = query_engine.get_prompts()
pprint.pprint(prompts_dict.keys())
pprint.pprint(prompts_dict["response_synthesizer:text_qa_template"].get_template())
"""
dict_keys(['response_synthesizer:text_qa_template', 'response_synthesizer:refine_template'])
('Context information is below.\n'
 '---------------------\n'
 '{context_str}\n'
 '---------------------\n'
 'Given the context information and not prior knowledge, answer the query.\n'
 'Query: {query_str}\n'
 'Answer: ')
"""

# 方法一：update_prompts修改默认的prompt(注意context_str和query_str两个变量名不能修改，否则在运行时绑定变量失败)
my_qa_prompt_template_str = (
    "以下是提供的上下文信息： \n"
    "--------------------------------\n"
    "{context_str}\n"
    "--------------------------------\n"
    "请根据上下文信息回答问题： \n"
    "{query_str}\n"
    "回答："
)


my_qa_prompt_template = PromptTemplate(template=my_qa_prompt_template_str)
query_engine.update_prompts(
    {
        "response_synthesizer:text_qa_template": my_qa_prompt_template,
    }
)

# 方法二: 组件初始化方法中直接修改query_engine的prompt, 和方法一效果相同，LlamaIndex中很多组件都可以使用这种方法
"""
query_engine = index.as_query_engine(
    text_qa_template=my_qa_prompt_template,
)
"""

# while True:
#     user_input = input("Enter a query: ")
#     if user_input.lower() == "exit":
#         break

#     response = query_engine.query(user_input)
#     print("Answer: ", response)
