from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.schema import MetadataMode, BaseNode, TransformComponent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.ingestion.pipeline import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from pathlib import Path
from dotenv import load_dotenv
import pprint
from typing import List
import re

load_dotenv(Path.cwd() / ".env")

llm = OpenAI(model="gpt-4o-mini")

docs = SimpleDirectoryReader(input_files=[Path.cwd() / "data/mcp.txt"]).load_data()


# 定义一个做数据清理的转换器
class TextCleaner(TransformComponent):
    def __call__(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        for node in nodes:
            node.text = re.sub(
                r"[^\u4e00-\u9fa5A-Za-z0-9，。！？‘’“”；：【】《》（）]", "", node.text
            )
        return nodes


# 构造一个数据摄取管道
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=500, chunk_overlap=0),
        TextCleaner(),
        TitleExtractor(llm=llm, show_progress=False),
    ]
)

nodes = pipeline.run(documents=docs)

pprint.pprint(nodes)
