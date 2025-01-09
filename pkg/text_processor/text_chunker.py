from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def pre_process_document(file_path: str):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=100, separators=["\n\n", "\n", " "]
    )
    return text_splitter.split_documents(documents)
