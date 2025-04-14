from llama_index.readers.web import SimpleWebPageReader

web_reader = SimpleWebPageReader(
    html_to_text=True,
)

docs = web_reader.load_data(urls=["https://www.google.com"])

print(docs)
