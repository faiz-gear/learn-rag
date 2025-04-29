from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

from dotenv import load_dotenv
import os
import pprint

load_dotenv(dotenv_path="../.env")

# 设置模型
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 设置回调
llm_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llm_debug])
Settings.callback_manager = callback_manager

# 构造 refine 响应生成器
from llama_index.core.schema import NodeWithScore
from llama_index.core.data_structs import Node
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer

refine_synthesizer = get_response_synthesizer(response_mode=ResponseMode.REFINE)

# 模拟检索出的3个Node
nodes = [
    NodeWithScore(
        node=Node(
            text="小麦手机是一款专为年轻人设计的手机，拥有时尚的外观和强大的性能。"
        ),
        score=0.9,
    ),
    NodeWithScore(
        node=Node(text="小麦手机采用了高通骁龙888处理器，拥有强大的性能。"), score=0.8
    ),
    NodeWithScore(
        node=Node(text="小麦手机还配备了5000mAh的大电池，续航能力非常强。"), score=0.7
    ),
]

# 使用 refine 响应生成器生成响应
response = refine_synthesizer.synthesize(
    "介绍一下小麦手机的优点，用中文回答", nodes=nodes
)
print(response)


def print_events_llm():
    events = llm_debug.get_event_pairs("llm")

    # 发生了多少次大模型的调用
    print(f"大模型调用次数: {len(events)}")

    # 依次打印大模型调用次的messages和response
    for i, event in enumerate(events):
        print(f"llm call {i+1} messages: {event[1].payload["messages"]}")
        pprint.pprint(event[1].payload["messages"])
        print(
            f"llm call {i+1} response: {event[1].payload["response"].message.content}"
        )
        pprint.pprint(event[1].payload["response"].message.content)


print_events_llm()
