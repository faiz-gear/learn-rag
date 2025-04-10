from llama_index.core.schema import Document
from llama_index.core import SimpleDirectoryReader
import pprint
import os

# 1.直接从字符串列表创建
texts = [
    "百度是一家中国互联网公司， 成立于2000年， 总部位于北京。",
    "阿里巴巴是一家中国电子商务公司， 成立于1999年， 总部位于杭州。",
    "腾讯是一家中国互联网公司， 成立于1998年， 总部位于深圳。",
]

docs = [Document(text=text) for text in texts]
pprint.pprint(docs)

# 2.从数据连接器加载数据生成
# 加载单个文件
docs2 = SimpleDirectoryReader(
    input_files=[os.path.join(os.path.dirname(__file__), "../../data/rust.pdf")]
).load_data()
print(len(docs2))  # 加载一个pdf文档，Documents数量是pdf的页数

# 加载目录
docs3 = SimpleDirectoryReader(
    input_dir=os.path.join(os.path.dirname(__file__), "../../data/llm_txt")
).load_data()
print(len(docs3))  # 加载一个目录，Documents数量是目录中所有文件的数量
