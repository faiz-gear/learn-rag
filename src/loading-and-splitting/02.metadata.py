from llama_index.core.schema import Document, MetadataMode

doc = Document(
    text="百度是一家中国互联网公司， 成立于2000年， 总部位于北京。",
    metadata={
        "filename": "test.txt",
        "author": "李彦宏",
        "category": "互联网公司",
    },
    excluded_llm_metadata_keys=[
        "filename"
    ],  # 在发送给大模型的的内容中，元数据的哪些key字段需要被排除
    excluded_embed_metadata_keys=[
        "filename",
        "author",
    ],  # 在生成向量时，元数据的哪些key字段需要被排除
    metadata_separator=" | ",  # 在构造元数据的字符串时，需要把元数据的多个key/value连接起来，这是定义的连接符
    metadata_template="{key}=>{value}",  # 每个元数据key/value的格式，默认为"{key}:{value}"
    text_template="Metadata: {metadata_str}\n-----\nContent: {content}",  # 元数据字符串和文本内容组合的格式，默认为"{metadata_str}\n\n{content}"
)

print(
    "\n全部元数据: \n",
    doc.get_content(metadata_mode=MetadataMode.ALL),
)

print(
    "\n发送给大模型的元数据: \n",
    doc.get_content(metadata_mode=MetadataMode.LLM),
)

print(
    "\n生成向量时使用的元数据: \n",
    doc.get_content(metadata_mode=MetadataMode.EMBED),
)

print(
    "\n没有元数据: \n",
    doc.get_content(metadata_mode=MetadataMode.NONE),
)
