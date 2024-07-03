# gradio_interface.py
import gradio as gr
import requests

def ask_local_server(question):
    url = "http://localhost:8000/ask"
    response = requests.post(url, json={"question": question})
    if response.status_code == 200:
        return response.json()["answer"]
    else:
        return "Error: Unable to get the response from the server."

iface = gr.Interface(
    fn=ask_local_server,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs=gr.Textbox(),
    title="Document Management System Q&A",
    description="Ask questions about the functionalities of the Document Management System."
)

iface.launch(share=True)
