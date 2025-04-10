from llama_index.core.schema import Document
import pprint

doc = Document(
    text="Hello, world!",
    metadata={
        "title": "RAG是一种常见的大模型应用范式， 它将文档转换为向量， 并使用向量检索技术来检索相关文档。",
        "author": "张三",
    },
)

pprint.pprint(doc.dict())

"""
text: 文档内容
metadata: 文档元数据
embedding: 文档向量
relationships: 类型是Dict, 保存相关的文档或Node信息。比如前后关系或父子关系
start_char_idx/end_char_idx: 文档内容在文本中的起始和结束位置
"""
