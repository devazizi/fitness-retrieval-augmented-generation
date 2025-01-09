import os
from google import genai
from fastapi import WebSocket


class GeminiHandler:
    __slot__ = ('api_key')

    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')

    async def create_embedded_text(self, texts: list):
        client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

        text_contents = [text.page_content for text in texts]

        response = await client.aio.models.embed_content(
            model='text-embedding-004',
            contents=text_contents,
        )

        return response.embeddings

    def retrival_content(self, context_text: str, question_text: str):
        text = ''
        prompt = f'''
            You are Health and Fitness AI assistant, Use the following context to answer the question. if question not about fitness, workout and health and relative context say question out of context, if you haven't enough information say so sorry i don't have enough information to answer the question.:

        Context:
        {context_text}

        Question:
            {question_text}
        '''

        client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        for response in client.models.generate_content_stream(
                model='gemini-2.0-flash-exp',
                contents=prompt
        ):
            text += response.text

            yield text
