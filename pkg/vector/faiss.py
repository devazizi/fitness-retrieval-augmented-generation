from langchain.vectorstores import FAISS
import faiss

import numpy as np
from pkg.openai.handler import OpenAIHandler
from typing import List
from langchain.embeddings.base import Embeddings


class CustomEmbeddings(Embeddings):
    def __init__(self):
        pass

    def embed_query(self, query: str) -> List[float]:
        embeded = OpenAIHandler().create_embedding(texts=[query])

        return embeded[0]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:

        return OpenAIHandler().create_embedding(texts=documents)


def index_to_docstore_id(index_id):
    return index_id


def create_vector_store(documents, embedding_func):
    texts = [doc.page_content for doc in documents]
    embeddings = embedding_func(texts)

    embeddings = np.array(embeddings).astype('float32')
    embedding_dim = len(embeddings[0])
    print(f"Embedding Dimension: {embedding_dim}")

    index = faiss.IndexFlatL2(embedding_dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    docstore = {i: doc for i, doc in enumerate(documents)}

    vector_db = FAISS(
        index=index,
        embedding_function=embedding_func,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    return vector_db


def save_vector_db(vector_db, file_path):
    vector_db.save_local(file_path)


def load_vector_db(file_path):
    custom_embeddings = CustomEmbeddings()

    return FAISS.load_local(file_path, custom_embeddings, allow_dangerous_deserialization=True)


def retrieve_docs(query, vector_db, embedding_func, top_k=6):
    query_embedding = embedding_func([query])
    query_embedding = np.array(query_embedding, dtype='float32')

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    D, I = vector_db.index.search(query_embedding, top_k)

    docs = [vector_db.docstore[i] for i in I[0]]

    return "\n\n".join([doc.page_content for doc in docs])
