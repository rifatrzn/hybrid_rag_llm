import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community import HuggingFaceEmbeddings
from langchain_core.runnables import chain
from dotenv import load_dotenv
import uvicorn
import gradio as gr
import requests
import threading
import subprocess
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize embeddings and load FAISS index
nvidia_api_key = os.getenv('NVIDIA_API_KEY')
if not nvidia_api_key:
    raise ValueError("NVIDIA_API_KEY environment variable is not set.")
os.environ['NVIDIA_API_KEY'] = nvidia_api_key

def load_faiss_index():
    try:
        # Initialize the embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            show_progress=True,
        )
        
        # Load the FAISS index
        vector = FAISS.load_local("./embed", embeddings, allow_dangerous_deserialization=True)
        retriever = vector.as_retriever()
        return vector, retriever
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index or initialize embeddings: {e}")

vector, retriever = load_faiss_index()

model = ChatNVIDIA( 
    model="meta/llama-3.1-70b-instruct",
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024
    )

# HYDE template and chain setup
hyde_template = """
Even if you do not know the full answer, generate a detailed and organic paragraph as a hypothetical response to the following question. Include relevant details and plausible scenarios that align with common functionalities in document management systems:

{question}

Example:
Q: How does the document management system handle user authentication?
A: The system uses a multi-layered approach to authentication, incorporating both password and biometric verification to ensure robust security. Users must first enter a secure password, followed by biometric verification such as a fingerprint or facial recognition. This dual-factor authentication helps protect sensitive data and prevents unauthorized access.
"""
hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
hyde_query_transformer = hyde_prompt | model | StrOutputParser()

@chain
def hyde_retriever(question):
    hypothetical_document = hyde_query_transformer.invoke({"question": question})
    return retriever.invoke(hypothetical_document)

template = """
Based on the following context, provide a detailed and precise answer to the question. Include relevant examples, explanations, and any available data to ensure clarity and depth. Directly address the question while enhancing the response with additional insights:

{context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
answer_chain = prompt | model | StrOutputParser()

@chain
def final_chain(question):
    documents = hyde_retriever.invoke(question)
    for s in answer_chain.stream({"question": question, "context": documents}):
        yield s

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    documents: List[dict]

class UpdateResponse(BaseModel):
    message: str
    output: str

app = FastAPI()

# Set up CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    question = request.question
    responses = []
    documents = []
    logger.info(f"Received question: {question}")
    try:
        retrieved_docs = hyde_retriever.invoke(question)
        for doc in retrieved_docs:
            documents.append({"content": doc.page_content, "source": doc.metadata.get("source", "Unknown")})
        
        for s in answer_chain.stream({"question": question, "context": retrieved_docs}):
            responses.append(s)
        
        logger.info(f"Returning response: {responses}")
        return AnswerResponse(answer="".join(responses), documents=documents)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

import subprocess

def is_update_needed():
    # Navigate to the cached repo and check for updates
    try:
        subprocess.run(["git", "fetch"], cwd="/app/cache/docs", check=True)
        local_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="/app/cache/docs").strip()
        remote_commit = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd="/app/cache/docs").strip()

        if local_commit != remote_commit:
            return True
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error checking for updates: {str(e)}")
        return False


@app.post("/update_db", response_model=UpdateResponse)
async def update_vector_db():
    try:
        # Trigger the update process
        update_vector_db_script()
        return UpdateResponse(message="Vector database updated successfully.", output="Update completed.")
    except Exception as e:
        logger.error(f"Error updating vector database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def update_vector_db_script():
    # This could directly call the shell script or include the logic within Python
    subprocess.run(["/bin/bash", "./fetch_and_build.sh"], check=True)

# Gradio interface
SYSTEM_NAME = "MIE Healthcare Management System"

def format_response(answer, documents):
    # Trim and format the response
    answer = answer.strip()
    formatted_answer = "\n".join(line.strip() for line in answer.splitlines() if line.strip())

    # Format document sources
    sources = "\n".join([f"- {doc['source']}" for doc in documents])
    if sources:
        formatted_answer += f"\n\n**Sources:**\n{sources}"
    
    return formatted_answer

def ask_local_server(message, history):
    url = "http://localhost:8000/ask"
    try:
        response = requests.post(url, json={"question": message})
        response.raise_for_status()
        answer = response.json()["answer"].strip()
        documents = response.json()["documents"]
        formatted_response = format_response(answer, documents)
        formatted_response = formatted_response.replace("{{% system-name %}}", SYSTEM_NAME)
        history.append((message, formatted_response))
        return history
    except requests.RequestException as e:
        error_message = f"Error: Unable to get the response from the server. {e}"
        history.append((message, error_message))
        return history


def update_db_local_server():
    url = "http://localhost:8000/update_db"
    try:
        response = requests.post(url)
        response.raise_for_status()
        output = response.json()["output"].strip()
        message = response.json()["message"]
        formatted_response = f"Update successful: {message}\n{output}".strip()
        return formatted_response
    except requests.RequestException as e:
        return f"Error: Unable to update the vector database. {e}"



# Gradio interface code remains the same
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # MIE Healthcare Management System Q&A
        ### Ask questions about the functionalities of the MIE Healthcare Management System.
        """
    )
    
    chatbot = gr.Chatbot()
    
    with gr.Row():
        question = gr.Textbox(
            lines=2,
            placeholder="Enter your question here...",
            show_label=False,
            elem_id="question-box"
        )
        ask_button = gr.Button("Send", elem_id="ask-button")
    
    with gr.Row():
        update_button = gr.Button("Update Vector Database", elem_id="update-button")
    
    def user_ask(user_message, history):
        return ask_local_server(user_message, history)

    def update_db(history):
        update_status = update_db_local_server()
        history.append(("", update_status))
        return history

    ask_button.click(fn=user_ask, inputs=[question, chatbot], outputs=chatbot)
    update_button.click(fn=update_db, inputs=[chatbot], outputs=chatbot)

def run_gradio_interface():
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    # Start the FastAPI server in a new thread
    threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host": "0.0.0.0", "port": 8000}).start()

    # Run the Gradio interface
    run_gradio_interface()