import os

from openai import Client


class OpenAIHandler:
    __slots__ = ('client',)

    def __init__(self):
        self.client = Client(
            api_key=os.getenv("OPEN_AI_EMBEDDING_KEY"),
            base_url=os.getenv("OPEN_AI_EMBEDDING_URL"),
        )

    def create_embedding(self, texts):
        embeddings = self.client.embeddings.create(
            model="BAAI/bge-m3",
            input=texts
        )

        return [item.embedding for item in embeddings.data]
