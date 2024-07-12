import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain
from dotenv import load_dotenv
import uvicorn
import gradio as gr
import requests
import threading


# Load environment variables
load_dotenv()

# Initialize embeddings and load FAISS index
nvidia_api_key = os.getenv('NVIDIA_API_KEY')
os.environ['NVIDIA_API_KEY'] = nvidia_api_key

embeddings = NVIDIAEmbeddings()
vector = FAISS.load_local("./FAISS_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector.as_retriever()



model = ChatNVIDIA(model="meta/llama3-70b-instruct")

# HYDE template and chain setup
hyde_template = """Even if you do not know the full answer, generate a one-paragraph hypothetical answer to the below question. Make sure to stay relevant to the topic and include plausible details that align with common functionalities in document management systems:

{question}"""
hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
hyde_query_transformer = hyde_prompt | model | StrOutputParser()

@chain
def hyde_retriever(question):
    hypothetical_document = hyde_query_transformer.invoke({"question": question})
    return retriever.invoke(hypothetical_document)

template = """Based on the following context, provide a concise answer to the question. Ensure your response includes relevant details, examples, and explanations to provide clarity and depth. Ensure your answer is precise, accurate, and directly addresses the question: 

Do not reply as according to context or provided context. Start with continuing fashion. 

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
    for s in final_chain.stream(question):
        responses.append(s)
    return AnswerResponse(answer="".join(responses))



# Gradio interface



SYSTEM_NAME = "MIE Healthcare Management System"

def ask_local_server(message, history):
    url = "http://localhost:8000/ask"
    response = requests.post(url, json={"question": message})
    if response.status_code == 200:
        answer = response.json()["answer"]
        answer = answer.replace("{{% system-name %}}", SYSTEM_NAME)
        history.append((message, answer))
        return history
    else:
        error_message = "Error: Unable to get the response from the server."
        history.append((message, error_message))
        return history


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
        ask_button = gr.Button("Submit", elem_id="ask-button")
    
    def user_ask(user_message, history):
        return ask_local_server(user_message, history)

    ask_button.click(fn=user_ask, inputs=[question, chatbot], outputs=chatbot)

def run_gradio_interface():
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    # Start the FastAPI server in a new thread
    threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host": "0.0.0.0", "port": 8000}).start()

    # Run the Gradio interface
    run_gradio_interface()