from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
import os

load_dotenv()

# 打印OpenAI API Key
print(os.getenv("OPENAI_API_KEY"))

# 加载文档
documents = SimpleDirectoryReader("data").load_data()

# 构造索引
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine()

# 查询
response = query_engine.query("Some question about the data should go here")
print(response)