# # 测试 Chroma
# import chromadb

# # 创建持久化客户端
# client = chromadb.PersistentClient(path="./chroma_db")

# # 测试心跳
# print(client.heartbeat())

import chromadb

chroma_client = chromadb.HttpClient(host="localhost", port=8000)

print(chroma_client.heartbeat())
