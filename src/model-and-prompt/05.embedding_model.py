from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# 默认的embedding模型
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

embeddings1 = embed_model.get_query_embedding("中国的首都是北京")

embeddings2 = embed_model.get_text_embedding("中国的首都是哪里")

embeddings3 = embed_model.get_text_embedding("苹果是一种好吃的水果")

print(embed_model.similarity(embeddings1, embeddings2))
print(embed_model.similarity(embeddings1, embeddings3))
