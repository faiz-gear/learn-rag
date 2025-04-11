from llama_index.core.schema import Document
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
import pprint
import os

# 手工设置元数据
doc1 = Document(
    text="AIGC是一种利用人工智能技术自动生成内容的方法，这些内容可以包括文本、图像、音频、视频等。\nAIGC的发展得益于深度学习的进步，特别是自然语言处理领域的成就, 使得计算能够更好的理解人类语言，从而生成更加符合人类语言习惯的内容。",
    metadata={
        "source": "AIGC",
        "file_name": "AIGC.txt",
        "category": "AI",
        "author": "AI",
        "date": "2024-01-01",
    },
)

print("\ndoc1\n")
pprint.pprint(doc1.metadata)

# 读取文件自动生成元数据
path = os.path.join(os.path.dirname(__file__), "../../data/mcp.txt")
doc2 = SimpleDirectoryReader(input_files=[path]).load_data()

print("\ndoc2\n")
pprint.pprint(doc2[0].metadata)

# 元数据自动继承到node
parser = TokenTextSplitter(chunk_size=100, chunk_overlap=0, separator="\n")
nodes = parser.get_nodes_from_documents(doc2)

print("\nnode\n")
pprint.pprint(nodes[0].metadata)
