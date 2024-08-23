import gradio as gr
import requests

SERVER_URL = "http://app:8001"

def api_request(endpoint, payload):
    url = f"{SERVER_URL}/{endpoint}"
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Error communicating with the server: {str(e)}"}

def update_vector_database():
    try:
        response = requests.post(f"{SERVER_URL}/update_vector_db")
        response.raise_for_status()
        result = response.json()
        return f"Success: {result['message']}\nDetails: {result['details']}"
    except requests.RequestException as e:
        return f"Failed to update vector database: {str(e)}"

def chat_with_model(message, history, provider, model, max_tokens):
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": msg} for i, (msg, _) in enumerate(history)]
    messages.append({"role": "user", "content": message})
    
    payload = {
        "provider": provider,
        "model": model,
        "prompt": message,
        "max_tokens": max_tokens,
        "history": messages
    }
    
    result = api_request("generate", payload)
    
    if result and "answer" in result:
        answer = result['answer']
        
        # Add evaluation metrics if available
        if "evaluation" in result:
            eval_metrics = result["evaluation"]
            answer += f"\n\n📊 Evaluation Metrics:"
            answer += f"\n   • Semantic Similarity: {eval_metrics['semantic_similarity']:.4f}"
            answer += f"\n   • Response Length: {eval_metrics['response_length']}"
        
        # Add related documents if available
        if "documents" in result:
            answer += "\n\n📚 Original Source:"
            for doc in result["documents"]:
                source = doc['metadata']['source'].split('/')[-1] if 'source' in doc['metadata'] else "Unknown"
                answer += f"\n   • {source}: {doc['content'][:100]}..."
        
        return answer
    elif result and "error" in result:
        return f"❌ Error: {result['error']}"
    else:
        return "❓ An unexpected error occurred."
    
    
css = """
.chatbot-container {
    border-radius: 10px;
    background-color: #f0f4f8;
}
.chatbot-container .message {
    padding: 10px;
    border-radius: 15px;
    margin-bottom: 10px;
}
.chatbot-container .user-message {
    background-color: #e3f2fd;
    text-align: right;
}
.chatbot-container .bot-message {
    background-color: #fff;
    border: 1px solid #e0e0e0;
}
.chatbot-container .bot-message pre {
    white-space: pre-wrap;
    word-wrap: break-word;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
        # 🏥 Healthcare Document Retrieval Q&A System 
        ### Ask questions about the functionalities of the Custom Healthcare Management System.
        """
    )
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(elem_classes="chatbot-container")
            msg = gr.Textbox(label="Your message 💬", placeholder="Type your message here...")
            send_btn = gr.Button("Send 🚀", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Settings")
            provider = gr.Radio(["ollama 🦙", "nvidia 🖥️"], label="Select provider", value="ollama 🦙")
            model = gr.Dropdown(["llama3.1 🦙", "llama3 🐪", "meta/llama-3.1-70b-instruct 🧠"], label="Select model", value="llama3.1 🦙")
            max_tokens = gr.Slider(minimum=1, maximum=500, value=100, step=1, label="Max Tokens 🔢", info="Adjust the length of the generated response")
            update_btn = gr.Button("Update Vector Database 🔄")
            update_output = gr.Textbox(label="Update Result 📊", lines=2)
               
            update_btn.click(
                fn=update_vector_database,  # Change this line
                outputs=update_output
            )

    def update_model_options(provider):
        if "ollama" in provider:
            return gr.Dropdown.update(choices=["llama3.1 🦙", "llama3 🐪"], value="llama3.1 🦙")
        else:
            return gr.Dropdown.update(choices=["meta/llama-3.1-70b-instruct 🧠"], value="meta/llama-3.1-70b-instruct 🧠")

    provider.change(update_model_options, inputs=[provider], outputs=[model])
    
    def respond(message, chat_history, provider, model, max_tokens):
        bot_message = chat_with_model(message, chat_history, provider.split()[0], model.split()[0], max_tokens)
        chat_history.append((f"👤 {message}", bot_message))
        return "", chat_history

    send_btn.click(respond, inputs=[msg, chatbot, provider, model, max_tokens], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot, provider, model, max_tokens], outputs=[msg, chatbot])
    update_btn.click(update_vector_database, outputs=[update_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)