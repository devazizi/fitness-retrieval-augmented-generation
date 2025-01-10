import os

from openai import Client, OpenAI


class OpenAIHandler:
    __slots__ = ('client',)

    def create_embedding(self, texts):
        self.client = Client(
            api_key=os.getenv("OPEN_AI_EMBEDDING_KEY"),
            base_url=os.getenv("OPEN_AI_EMBEDDING_URL"),
        )

        embeddings = self.client.embeddings.create(
            model="BAAI/bge-m3",
            input=texts
        )

        return [item.embedding for item in embeddings.data]

    async def text_generator(self, context_text, question_text):
        client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama', # required, but unused
            # api_key='',
            # base_url='http://localhost:11434/api',
        )
        print(client.base_url)

        prompt = f'''
            You are Health and Fitness AI assistant, Use the following context to answer the question. if question not about fitness, workout and health and relative context say question out of context, if you haven't enough information say so sorry i don't have enough information to answer the question.:

        Context:
        {context_text}

        Question:
            {question_text}
        '''

        try:
            for stream_chunk in client.completions.create(
                model='phi3:mini',
                prompt=prompt,
                # messages=[
                #     {"role": "system", "content": "You are a helpful assistant. Your name is FitHelper"},
                #     {"role": "user", "content": prompt},
                # ],
                max_tokens=300,
                stream=True
            ):
                delta_content = stream_chunk.choices[0].text
                if delta_content:
                    yield delta_content

        except Exception as e:
            print(f"Error occurred: {e.__str__()}")
            yield "sorry system already crrupt"
