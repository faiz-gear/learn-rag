from llama_index.core import DocumentSummaryIndex, Document
from pprint import pprint

documents = [
    Document(
        text="百度是一家中国互联网公司，成立于2000年，总部位于北京。",
        metadata={"source": "百度官网"},
        doc_id="1",
    ),
    Document(
        text="阿里巴巴是一家中国电子商务公司，成立于1999年，总部位于杭州。",
        metadata={"source": "阿里巴巴官网"},
        doc_id="2",
    ),
    Document(
        text="腾讯是一家中国互联网公司，成立于1998年，总部位于深圳。",
        metadata={"source": "腾讯官网"},
        doc_id="3",
    ),
]

# storage_context: 设置向量库等
# summary_query: 设置摘要生成提示
# llm: 设置LLM模型
# transformations: 设置数据摄取需要的转换器
document_summary_index = DocumentSummaryIndex.from_documents(
    documents,
    summary_query="用中文描述所给文本的主要内容，同时描述这段文本可以回答的问题",
)

pprint(document_summary_index.get_document_summary("1"))

query_engine = document_summary_index.as_query_engine()

response = query_engine.query("百度是一家中国互联网公司吗？")

print(response)
