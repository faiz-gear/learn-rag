from llama_index.readers.database import DatabaseReader
from llama_index.core import Document
from llama_index.core.schema import MetadataMode
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))


def print_docs(docs: list[Document]):
    print("Count of documents: ", len(docs))
    for index, doc in enumerate(docs):
        print("-" * 10)
        print(f"Document {index}:")
        print(doc.get_content(metadata_mode=MetadataMode.ALL))
        print("-" * 100)


print("db uri: ", os.getenv("PGURI"))

db = DatabaseReader(
    uri=os.getenv("PGURI"),
)

docs = db.load_data(
    query="SELECT * FROM unicorns LIMIT 10",
)

print_docs(docs)
