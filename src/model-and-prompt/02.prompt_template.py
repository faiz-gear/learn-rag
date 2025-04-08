from llama_index.core import PromptTemplate

template = (
    "以下是提供的上下文信息： \n"
    "{context}\n"
    "请根据上下文信息回答问题： \n"
    "{query}"
)

qa_template = PromptTemplate(template=template)

print(
    qa_template.format(
        context="小麦15 PRO是小麦公司最新推出的6.7存大屏旗舰手机",
        query="小麦15 PRO的屏幕尺寸是多少？",
    )
)


print(
    qa_template.format_messages(
        context="小麦15 PRO是小麦公司最新推出的6.7存大屏旗舰手机",
        query="小麦15 PRO的屏幕尺寸是多少？",
    )
)
