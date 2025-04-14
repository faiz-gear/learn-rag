from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.schema import MetadataMode
from llama_index.core.readers.base import BaseReader
from typing import Any, Optional, Dict
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))


def print_docs(docs: list[Document]):
    print("Count of documents: ", len(docs))
    for index, doc in enumerate(docs):
        print("-" * 10)
        print(f"Document {index}:")
        print(doc.get_content(metadata_mode=MetadataMode.ALL))
        print("-" * 100)


class PSQLReader(BaseReader):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> list[Document]:
        with open(file) as f:
            content = f.read()

            # 执行文档中的SQL, 获取SQL执行结果
            result = execute_sql(content)
            print("result: ", result)

            metadata = {"file_suffix": "psql"}

            if extra_info:
                metadata = {**metadata, **extra_info}

            return [Document(text=result, metadata=metadata)]


def execute_sql(sql: str) -> str:
    # 执行SQL, 获取执行结果
    print("pg host: ", os.getenv("PGHOST"))
    print("pg database: ", os.getenv("PGDATABASE"))
    print("pg user: ", os.getenv("PGUSER"))
    print("pg password: ", os.getenv("PGPASSWORD"))
    conn = psycopg2.connect(
        host=os.getenv("PGHOST"),
        database=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
    )
    cur = conn.cursor()

    cur.execute(sql)

    results = []
    for result in cur:
        results.append(str(result))

    conn.close()
    return "\n".join(results)


reader = SimpleDirectoryReader(
    input_files=[os.path.join(os.path.dirname(__file__), "../../data/test.psql")],
    file_extractor={".psql": PSQLReader()},
)

docs = reader.load_data()
print_docs(docs)
