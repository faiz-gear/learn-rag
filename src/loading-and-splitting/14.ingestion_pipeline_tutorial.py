from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.schema import MetadataMode, BaseNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion.pipeline import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.vector_stores import VectorStoreQuery
from pathlib import Path
from dotenv import load_dotenv
import pprint
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
import time

load_dotenv(Path.cwd() / ".env")

llm = OpenAI(model="gpt-4o-mini")

embed_model = OpenAIEmbedding(model="text-embedding-3-small")

docs = SimpleDirectoryReader(input_files=[Path.cwd() / "data/mcp.txt"]).load_data()

# 构造一个向量存储对象存储最后存储的Node对象
chroma = chromadb.PersistentClient(path="./chroma_db")

# try:
#     if chroma.get_collection(name="ragdb"):
#         chroma.delete_collection(name="ragdb")
# except Exception as e:
#     print(f"Error deleting collection: {e}")

collection = chroma.get_or_create_collection(
    name="pipeline_tutorial", metadata={"hnsw:space": "cosine"}
)
chroma_vector_store = ChromaVectorStore(
    chroma_collection=collection,
)

if __name__ == "__main__":
    # 构造一个数据摄取管道
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=500, chunk_overlap=0),
            TitleExtractor(llm=llm, show_progress=False),
            # 接收nodes并进行向量化
            embed_model,
        ],
        # 在数据管道运行完成后，会将向量化后的node列表添加的向量库中
        vector_store=chroma_vector_store,
    )
    pipeline.load("./pipeline_storage")

    time_start = time.time()
    # 设置num_workers开启多进程并行处理摄取管道的任务
    nodes = pipeline.run(documents=docs, num_workers=1, show_progress=True)
    time_end = time.time()
    print(f"数据摄取管道运行时间: {time_end - time_start} 秒")

    print(f"{len(nodes)} nodes ingested")

    # pprint.pprint(nodes)

    # results = chroma_vector_store.query(
    #     VectorStoreQuery(query_str="什么是MCP?", similarity_top_k=1)
    # )

    # pprint.pprint(results.nodes)

    # 本地文档存储，在下一次运行摄取管道时，对比从文档存储对象读取的文档信息与本次输入的文档信息，使用没有发生变化的部分（hash对比）
    pipeline.persist("./pipeline_storage")
