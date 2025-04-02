import chromadb
from langchain_community.llms.openai import OpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")

llm = OpenAI(model="gpt-4o-mini")
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")


# 加载与读取文档

loader = DirectoryLoader(
    os.path.join(os.path.dirname(__file__), "../data/"),
    glob="*.txt",
    loader_cls=TextLoader,
)
documents = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

# 准备向量存储
chroma = chromadb.HttpClient(host="localhost", port=8000)
# chroma.delete_collection(name="ragdb")
collection = chroma.get_or_create_collection(
    name="ragdb", metadata={"hnsw:space": "cosine"}
)
db = Chroma(
    client=chroma,
    collection_name="ragdb",
    embedding_function=embed_model,
)

# 创建向量存储索引
db.add_documents(texts)

# 构造查询引擎
retriever = db.as_retriever()

# 构造一个RAG链
prompt = hub.pull("rlm/rag-prompt")
rag_chain = (
    {
        "context": retriever
        | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


while True:
    user_input = input("Enter a query: ")
    if user_input.lower() == "exit":
        break

    response = rag_chain.invoke(user_input)
    print("Answer: ", response)
