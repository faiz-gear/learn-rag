from llama_index.core import VectorStoreIndex, Document


documents = [
    Document(
        text="百度是一家中国互联网公司，成立于2000年，总部位于北京。",
        metadata={"source": "百度官网"},
    ),
    Document(
        text="阿里巴巴是一家中国电子商务公司，成立于1999年，总部位于杭州。",
        metadata={"source": "阿里巴巴官网"},
    ),
    Document(
        text="腾讯是一家中国互联网公司，成立于1998年，总部位于深圳。",
        metadata={"source": "腾讯官网"},
    ),
]

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("百度是一家中国互联网公司吗？")
print(response)
