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

# 1.如果需要修改模板的变量名称为更有意义的变量名称，可以使用template_var_mappings参数
# my_qa_prompt_template_str = (
#     "以下是提供的上下文信息： \n"
#     "--------------------------------\n"
#     "{my_context_str}\n"
#     "--------------------------------\n"
#     "请根据上下文信息回答问题： \n"
#     "{my_query_str}\n"
#     "回答："
# )

# template_var_mappings = {
#     "my_context_str": "context_str",
#     "my_query_str": "query_str",
# }

# my_qa_prompt_template = PromptTemplate(
#     template=my_qa_prompt_template_str, template_var_mappings=template_var_mappings
# )

# 2.也可以通过function_mappings来更灵活的设置变量
my_qa_prompt_template_str = (
    "以下是提供的上下文信息： \n"
    "--------------------------------\n"
    "{context_str}\n"
    "--------------------------------\n"
    "请根据上下文信息回答问题： \n"
    "{query_str}\n"
    "回答："
)


def fn_context_str(**kwargs) -> str:
    # 自定义context_str的获取方式
    return kwargs["context_str"] + " 自定义context_str的获取方式"


def fn_query_str(**kwargs) -> str:
    # 自定义query_str的获取方式
    return kwargs["query_str"] + " 自定义query_str的获取方式"


my_qa_prompt_template = PromptTemplate(
    template=my_qa_prompt_template_str,
    function_mappings={
        "context_str": fn_context_str,
        "query_str": fn_query_str,
    },
)

pprint.pprint(
    my_qa_prompt_template.format(
        context_str="小麦15 PRO是小麦公司最新推出的6.7存大屏旗舰手机",
        query_str="小麦15 PRO的屏幕尺寸是多少？",
    )
)

query_engine.update_prompts(
    {
        "response_synthesizer:text_qa_template": my_qa_prompt_template,
    }
)


# while True:
#     user_input = input("Enter a query: ")
#     if user_input.lower() == "exit":
#         break

#     response = query_engine.query(user_input)
#     print("Answer: ", response)
