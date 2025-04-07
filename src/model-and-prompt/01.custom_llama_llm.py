from typing import Any

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback


class MyLLM(CustomLLM):
    """
    自定义LLM类
    """

    model_name: str = "my_llm"
    dummy_response: str = "您好，我是一个正在开发中的LLM模型，我还没有学会回答问题。"

    # 实现metadata接口
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model_name,
        )

    # 实现complete接口
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(
            text=self.dummy_response,
        )

    # 实现stream_complete接口
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response = ""
        for chunk in self.dummy_response:
            response += chunk
            yield CompletionResponse(
                text=response,
                delta=chunk,
            )


llm = MyLLM()
print(llm.complete("你好"))

for chunk in llm.stream_complete("你好"):
    print(chunk)


"""
全局设置大模型组件

Settings.llm = MyLLM()
"""


"""
插入其他模块中

from llama_index.core.indices.keyword_table import KeywordTableIndex

KeywordTableIndex.from_documents(documents, llm = MyLLM())
"""
