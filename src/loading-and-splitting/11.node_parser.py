import re
from llama_index.core.node_parser import (
    TokenTextSplitter,
    SentenceSplitter,
    SentenceWindowNodeParser,
    HierarchicalNodeParser,
    SemanticSplitterNodeParser,
    SimpleFileNodeParser,
)
from llama_index.core import Document
from llama_index.core.schema import MetadataMode, BaseNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import FlatReader
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))


def print_nodes(nodes: list[BaseNode]):
    print("Count of nodes: ", len(nodes))
    for index, node in enumerate(nodes):
        print("-" * 10)
        print(f"Node {index}:")
        print(f"Node: {node}")
        print(f"Node relationships: {node.relationships}")
        print(f"Node content: {node.get_content(metadata_mode=MetadataMode.ALL)}")
        print("-" * 100)


"""
无论什么类型的文本分割器，都是基于以下4种基础分割方法:
1.分隔符分割 split_by_sep
2.句子分割 split_by_sentence_tokenizer
3.正则表达式分割 split_by_regex
4.字符分割 split_by_char 一般不使用，会丢失语义信息


"""

# 1.TokenTextSplitter 最基础的数据分割器

docs = [
    Document(
        text="Google公司介绍\nGoogle是一家全球知名的互联网公司，成立于1998年\n总部位于美国加利福尼亚州山景城。主要产品是搜索引擎，包括网页搜索、图片搜索、视频搜索等。"
    )
]

# 使用TokenTextSplitter进行分割
"""
chunk_size: 分割出的文本块（Node对象的文本内容）的Token数量大小，默认用的OpenAI的tiktoken库
chunk_overlap: 分割出的文本块之间的重叠的Token数量大小
a.这两个参数只是上限参数，实际分割出的文本块的Token数量会小于或等于这个值
b.默认chunk_size会包含元数据的大小，因此实际的文本内容会比chunk_size小
c.分割器会尽量确保分割出的文本块的大小接近chunk_size，但是不会超过chunk_size
"""

"""
分割方法,按优先级使用以下3中基础分割方法:
1.分隔符分割 split_by_sep（基于分隔符参数）
2.分隔符分割 split_by_sep（基于backup_separators参数）
3.基于字符分割 split_by_char
"""
token_splitter = TokenTextSplitter(
    chunk_size=50, chunk_overlap=0, separator="\n", backup_separators=["。"]
)

# nodes = token_splitter.get_nodes_from_documents(docs)
# print_nodes(nodes)

# 由于按照\n分割， 且每个文本块的大小不超过并接近50， 因此输出结果中前两个文本块进行了合并


# 2.SentenceSplitter 基于段落与句子分割的文本分割器


# 自定义分割文本的函数
def my_chunking_tokenizer_fn(text: str) -> list[str]:
    """
    自定义分割文本的函数
    """
    print("start my_chunking_tokenizer_fn")
    sentence_delimiters = re.compile("[。！？]")
    sentences = sentence_delimiters.split(text)
    print("sentences: ", sentences)
    return [s.strip() for s in sentences if s.strip()]


docs_sentence = [
    Document(
        text="***Google公司介绍***Google是一家全球知名的互联网公司，成立于1998年***总部位于美国加利福尼亚州山景城xxxxxxxxxxxxxxxxxxxxxx。主要产品是搜索引擎，包括网页搜索、图片搜索、视频搜索等。"
    )
]


"""
secondary_chunking_regex这个正则表达式会匹配'非标点符号的文本 + 可选的一个标点符号'
例如对于文本 'Hello, world. 你好。' 会被分割成：
'Hello,'
'world.'
'你好。'
"""


"""
分割方法,按优先级使用以下5种基础分割方法:
1.基于分隔符分割（基于paragraph_separator参数，默认为"\n\n\n"）
2.基于句子分割（可通过chunking_tokenizer_fn参数自定义， 默认为使用nltk库的句子分割器）
3.基于正则表达式分割（可通过secondary_chunking_regex参数自定义, 默认为r"[^,.;。！？]+[,.;。！？]?"）
4.基于分隔符分割（可通过separator参数自定义, 默认为"\n"）
5.基于字符分割（基本不使用）
"""
sentence_splitter = SentenceSplitter(
    chunk_size=50,
    chunk_overlap=0,
    paragraph_separator="。",
    chunking_tokenizer_fn=my_chunking_tokenizer_fn,
    secondary_chunking_regex=r"[^,.;。！？]+[,.;。！？]?",
    separator="\n",
)

# nodes = sentence_splitter.get_nodes_from_documents(docs_sentence)
# print_nodes(nodes)

"""
由于基于paragraph_separator分隔，无法把文本内容都分割成token大小小于50的文本块，因此会使用chunking_tokenizer_fn进行分割
"""


# 3.SentenceWindowNodeParser

"""
SentenceWindowNodeParser是SentenceSplitter的改进版，它使用滑动窗口的方式来分割文本。
与其他元数据不一样，这个window的内容默认是对嵌入模型或者大模型不可见的
不建议把window的内容输入到嵌入模型，嵌入应该只针对本Node对象的文本内容，这样有利于语义的戏份，可以提供后面检索的精确度
建议把node内容替换成window包含的内容发送到大模型用于生成，以帮助大模型获得更多的上下文，提高生成质量
"""

docs_sentence_window = [
    Document(
        text="Google公司介绍介绍：Google是一家全球知名的互联网公司。\
              总部位于美国加利福尼亚州山景城\
              主要产品是搜索引擎，包括网页搜索、图片搜索、视频搜索等。\
              百度是Google的竞争对手。"
    )
]

sentence_window_splitter = SentenceWindowNodeParser(
    window_size=2, sentence_splitter=my_chunking_tokenizer_fn  # 包含几个句子
)

# nodes = sentence_window_splitter.get_nodes_from_documents(docs_sentence_window)
# print_nodes(nodes)

# 4.HierarchyNodeParser

"""
HierarchyNodeParser是SentenceWindowNodeParser的改进版，它使用层次结构的方式来分割文本, 生成具备层次关系的Node节点，每个Node对象都可以找到它的父节点和子节点
chunk_sizes参数是一个列表，列表中的每个元素表示一个层次的Node对象的文本块的大小
大文本块的Node对象可以包含多个小文本块的Node对象作为子节点
"""

docs_hierarchy = [
    Document(
        text="Google公司介绍介绍：Google是一家全球知名的互联网公司。\
              总部位于美国加利福尼亚州山景城\
              主要产品是搜索引擎，包括网页搜索、图片搜索、视频搜索等。\
              百度是Google的竞争对手。\
              百度公司成立于1998年，总部位于中国北京。"
    )
]

hierarchy_splitter = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 100, 50])

# nodes = hierarchy_splitter.get_nodes_from_documents(docs_hierarchy)
# print_nodes(nodes)

# 5.SemanticSplitterNodeParser

"""
SemanticSplitterNodeParser是基于语义与向量的文本分割器，它借助嵌入模型来识别不同句子的语义相似度，并进行合并，从而确保语义上更相关的句子被合并到一起
先根据sentence_splitter进行分割，然后根据embed_model计算每个句子的向量，然后根据breakpoint_percentile_threshold参数设置的阈值，将语义上更相关的句子合并到一起
"""

docs_semantic = [
    Document(
        text="Google公司介绍介绍：Google是一家全球知名的互联网公司。\
              总部位于美国加利福尼亚州山景城\
              主要产品是搜索引擎，包括网页搜索、图片搜索、视频搜索等。\
              百度是Google的竞争对手。\
              百度公司成立于1998年，总部位于中国北京。"
    )
]


embed_model = OpenAIEmbedding(model="text-embedding-3-small")

semantic_splitter = SemanticSplitterNodeParser.from_defaults(
    breakpoint_percentile_threshold=20,  # 越大，语义上越相关的句子越容易被合并到一起，生成的node对象越少
    embed_model=embed_model,
    sentence_splitter=my_chunking_tokenizer_fn,
)

# nodes = semantic_splitter.get_nodes_from_documents(docs_semantic)
# print_nodes(nodes)

# 6.SimpleFileNodeParser

"""
SimpleFileNodeParser是基于文件的文本分割器，它使用文件的结构来分割文本。
"""


docs_simple_file = FlatReader().load_data(Path.cwd() / "data" / "mcp.txt")

simple_file_splitter = SimpleFileNodeParser.from_defaults()

nodes = simple_file_splitter.get_nodes_from_documents(docs_simple_file)
print_nodes(nodes)
