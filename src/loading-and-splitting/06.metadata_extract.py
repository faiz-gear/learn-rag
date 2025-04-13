from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.core.schema import MetadataMode
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
import os
import pprint

load_dotenv(dotenv_path="../../.env")

docs = SimpleDirectoryReader(
    input_files=[os.path.join(os.path.dirname(__file__), "../../data/mcp.txt")],
).load_data()

# print(docs)

nodes = SentenceSplitter(
    chunk_size=100, chunk_overlap=0, separator="\n"
).get_nodes_from_documents(docs)

# automatically extracts a summary over a set of Nodes
# summary_extractor = SummaryExtractor(
#     llm=OpenAI(model="gpt-4o-mini"),
#     show_progress=True,
#     prompt_template="""
#     请生成以下内容的中文摘要：
#     {context_str}
# 		\n 摘要：
#     """,
#     metadata_mode=MetadataMode.NONE,  # 元数据输入模式，要求在调用大模型生成摘要时，不输入元数据
# )

# summary = summary_extractor.extract(nodes)

# print(summary)

# extracts a set of questions that each Node can answer
# question_extractor = QuestionsAnsweredExtractor(
#     llm=OpenAI(model="gpt-4o-mini"),
#     show_progress=True,
#     metadata_mode=MetadataMode.NONE,  # 元数据输入模式，要求在调用大模型生成问题时，不输入元数据
# )

# questions = question_extractor.extract(nodes)

# pprint.pprint(questions)


# extracts a title over the context of each Node
# title_extractor = TitleExtractor(
#     llm=OpenAI(model="gpt-4o-mini"),
#     show_progress=True,
#     metadata_mode=MetadataMode.NONE,  # 元数据输入模式，要求在调用大模型生成标题时，不输入元数据
# )

# # 抽取器生成的标题是文档级别的
# titles = title_extractor.extract(nodes)

# pprint.pprint(titles)
