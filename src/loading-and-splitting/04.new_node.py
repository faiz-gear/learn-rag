from llama_index.core.schema import (
    TextNode,
    Document,
    NodeRelationship,
    RelatedNodeInfo,
)
from llama_index.core.node_parser import TokenTextSplitter
import pprint

# 1.直接生成
texts = [
    "百度是一家中国互联网公司， 成立于2000年， 总部位于北京。",
    "阿里巴巴是一家中国电子商务公司， 成立于1999年， 总部位于杭州。",
    "腾讯是一家中国互联网公司， 成立于1998年， 总部位于深圳。",
]

nodes = [TextNode(text=text) for text in texts]
print("直接生成\n")
pprint.pprint(nodes)

# 直接生成的Node对象和使用Document对象几乎一样，因为Document集成自TextNode

# 2.从Document生成
docs = [
    Document(
        text="AIGC是一种利用人工智能技术自动生成内容的方法，这些内容可以包括文本、图像、音频、视频等。\n AIGC的发展得益于深度学习的进步，特别是自然语言处理领域的成就, 使得计算能够更好的理解人类语言，从而生成更加符合人类语言习惯的内容。"
    )
]

print("\nDocument对象\n")
pprint.pprint(docs)

parser = TokenTextSplitter(chunk_size=100, chunk_overlap=2, separator="\n")
nodes = parser.get_nodes_from_documents(docs)
print("\n从Document生成\n")
pprint.pprint(nodes)
for i, node in enumerate(nodes):
    print(f"Node {i+1}:\n{node.text}\n")


# Node对象之间的关系
print("\nNode 0 relationships\n")
pprint.pprint(nodes[0].relationships)

print("\nNode 1 relationships\n")
pprint.pprint(nodes[1].relationships)

# 手动设置node对象之间的关系
custom_node = TextNode(text="custom node")
print("\ncustom node\n")
pprint.pprint(custom_node)
nodes[0].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
    node_id=custom_node.node_id,
)
print("\nNode 0 relationships\n")
pprint.pprint(nodes[0].relationships)
