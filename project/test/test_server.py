from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.embeddings import OllamaEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

app = FastAPI()

# CORS middleware setup (unchanged)

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 100  # Default value

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    
# class ChatRequest(BaseModel):
#     provider: str
#     model: str
#     messages: List[Dict[str, str]]
#     base_url: str = "http://localhost:11434"

@app.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
    result = embeddings.embed_query(request.text)
    return EmbeddingResponse(embedding=result)

@app.post("/generate")
async def generate(request: GenerateRequest):
    llm = Ollama(base_url="http://localhost:11434", model=request.model, callbacks=[StreamingStdOutCallbackHandler()])
    
    def generate_stream():
        for chunk in llm.stream(request.prompt, max_tokens=request.max_tokens):
            yield f"data: {chunk}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")

@app.post("/chat")
async def chat(request: ChatRequest):
    llm = Ollama(base_url="http://localhost:11434", model=request.model, callbacks=[StreamingStdOutCallbackHandler()])
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        *[HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in request.messages]
    ]
    
    def chat_stream():
        for chunk in llm.stream(messages):
            yield f"data: {chunk}\n\n"
    
    return StreamingResponse(chat_stream(), media_type="text/event-stream")


# @app.post("/chat")
# async def chat(request: ChatRequest):
#     try:
#         if request.provider.lower() == "ollama":
#             last_message = request.messages[-1]['content'] if request.messages else ""

#             # Retrieve relevant documents based on the last user message
#             retrieved_docs = retriever.get_relevant_documents(last_message)
#             context = "\n\n".join([doc.page_content for doc in retrieved_docs])

#             # Update conversation memory with key points from the context
#             conversation_memory.add_memory(context)

#             # Combine memory and context for generating response
#             memory_context = conversation_memory.get_memory()
#             combined_context = f"{memory_context}\n\n{context}" if context not in memory_context else memory_context

#             # Prepare the chat payload with the combined context and memory
#             chat_messages = request.messages
#             if combined_context.strip():
#                 chat_messages.insert(0, {"role": "system", "content": f"Context:\n{combined_context}"})

#             payload = {
#                 "model": request.model,
#                 "messages": chat_messages
#             }

#             result = call_ollama_api("/api/chat", payload)
#             answer = "".join([res['response'] for res in result])

#             # Add the latest response to the memory
#             conversation_memory.add_memory(answer)

#             return {
#                 "answer": answer,
#                 "documents": [
#                     {"content": doc.page_content[:150], "metadata": doc.metadata}
#                     for doc in retrieved_docs[:3]
#                 ]
#             }
#         else:
#             # Use the existing LLM and retrieval chain for non-Ollama providers
#             llm = get_llm(request.provider, request.model)
#             qa_chain = get_qa_chain(llm, retriever)
#             last_message = request.messages[-1]['content'] if request.messages else ""
#             result = qa_chain.invoke({"query": last_message})

#             # Update conversation memory with key points from the response
#             conversation_memory.add_memory(result['result'])

#             return {
#                 "answer": result['result'],
#                 "documents": [
#                     {"content": doc.page_content[:150], "metadata": doc.metadata}
#                     for doc in result['source_documents'][:3]
#                 ]
#             }
#     except Exception as e:
#         logger.error(f"Error in chat: {str(e)}")
#         if "Connection refused" in str(e):
#             error_message = "Failed to connect to Ollama. Please ensure Ollama is running."
#         elif "Unauthorized" in str(e):
#             error_message = "NVIDIA API authentication failed. Please check your API key."
#         else:
#             error_message = str(e)
#         raise HTTPException(status_code=500, detail=error_message)







if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)