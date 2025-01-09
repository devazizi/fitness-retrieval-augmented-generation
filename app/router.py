# from fastapi.routing import APIRouter
# from starlette.websockets import WebSocket, WebSocketDisconnect
# from .service import GeminiHandler
#
# rag_router = APIRouter()
#
# @rag_router.websocket('/chat')
# async def chat(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             query = await websocket.receive_text()
#
#             path='/home/ali/Documents/development/projects/simple-retrieval-augmented-generation/vectordb'
#             emb = ProcessDocument().pre_process_document(query)
#
#             vector_db = await ProcessDocument().load_vector_db(path, emb)
#             print(await vector_db.asimilarity_search(query=query))
#             # vector_db.similarity_search(query)
#             # context = ProcessDocument().retrieve_docs(query, vector_db)
#             # await websocket.send_text("**Answer:**\n")
#             # await GeminiHandler().retrival_content(context, query, websocket)
#             # await websocket.send_text(f"\n\n**Context Used:**\n{context}")
#     except WebSocketDisconnect:
#         print("Client disconnected.")