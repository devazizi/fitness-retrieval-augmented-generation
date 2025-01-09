import os
from dotenv import load_dotenv
from fastapi import FastAPI

from pkg.vector.faiss import load_vector_db, create_vector_store, save_vector_db, retrieve_docs
from pkg.text_processor.text_chunker import pre_process_document
from pkg.openai.handler import OpenAIHandler
from pkg.gemini.handler import GeminiHandler
import time
import gradio as gr


load_dotenv()

VECTOR_DB_PATH = "vector_store"
DOCUMENT_PATH = "/knowledge_example.txt"


if os.path.exists(VECTOR_DB_PATH):
    vector_db = load_vector_db(VECTOR_DB_PATH)
else:
    print("Creating vector database...")
    docs = pre_process_document(os.getcwd() + DOCUMENT_PATH)
    vector_db = create_vector_store(docs, OpenAIHandler().create_embedding)
    save_vector_db(vector_db, VECTOR_DB_PATH)
    print("Vector database created and saved!")

def generate_response(message, history):

    context = retrieve_docs(message, vector_db, OpenAIHandler().create_embedding)

    for data in GeminiHandler().retrival_content(context, message):
        yield data


gr.ChatInterface(
    fn=generate_response,
    type="messages",
    examples=["what is fitness?", 'How to lose fat?'],
    title="Fitness Helper",
    description="Ask any Question about Fitness",
    theme="ocean",
).launch()

# app = FastAPI()



# app.include_router(rag_router)




