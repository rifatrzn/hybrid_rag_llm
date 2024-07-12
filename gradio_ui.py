import gradio as gr
import requests


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

# Gradio interface
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

demo.launch(share=True)
