from llama_index.readers.web import BeautifulSoupWebReader
from typing import Any, Tuple, Dict
from llama_index.core import Document
from llama_index.core.schema import MetadataMode


def print_docs(docs: list[Document]):
    print("Count of documents: ", len(docs))
    for index, doc in enumerate(docs):
        print("-" * 10)
        print(f"Document {index}:")
        print(doc.get_content(metadata_mode=MetadataMode.ALL))
        print("-" * 100)


# 定义一个个性化的网页内容提取方式
def _baidu_reader(
    soup: Any, url: str, include_url_in_text: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """
    百度网页内容提取方式
    """
    main_content = soup.find(class_="main-content")
    if main_content:
        text = main_content.get_text()
    else:
        text = ""
    return text, {"title": soup.find(class_="post__title").get_text()}


web_reader = BeautifulSoupWebReader(
    website_extractor={"cloud.baidu.com": _baidu_reader}
)

docs = web_reader.load_data(
    urls=["https://cloud.baidu.com/doc/WENXINWORKSHOP/s/7ltgucw50"]
)

print_docs(docs)
