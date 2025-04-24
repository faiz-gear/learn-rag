from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleObjectNodeMapping

obj1 = {
    "name": "小米",
    "cpu": "骁龙888",
    "gpu": "Adreno 660",
    "ram": "8GB",
    "storage": "128GB",
}
obj2 = ["Iphone", "小米", "华为", "三星", "OPPO", "VIVO"]
obj3 = (["A", "B", "C"], {"a": 1, "b": 2, "c": 3})
obj4 = "大模型是一种基于大量数据训练的神经网络模型，可以处理自然语言等复杂任务。"

arbitrary_objects = [obj1, obj2, obj3]

# (optional) object-node mapping
obj_node_mapping = SimpleObjectNodeMapping.from_objects(arbitrary_objects)
nodes = obj_node_mapping.to_nodes(arbitrary_objects)

# object index
object_index = ObjectIndex(
    index=VectorStoreIndex(nodes=nodes),
    object_node_mapping=obj_node_mapping,
)

# object index from_objects (default index_cls=VectorStoreIndex)
# object_index = ObjectIndex.from_objects(arbitrary_objects, index_cls=VectorStoreIndex)

object_retriever = object_index.as_retriever()
response = object_retriever.retrieve("小米手机的cpu是什么？")
print(response)
