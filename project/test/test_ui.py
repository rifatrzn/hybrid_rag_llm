import streamlit as st
import requests
import json

def get_embeddings(text):
    url = "http://localhost:8000/embeddings"
    response = requests.post(url, json={"text": text})
    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        return f"Error: {response.status_code}, {response.text}"

def generate_response(prompt, model="llama3.1", max_tokens=100):
    url = "http://localhost:8000/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    try:
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        yield line[6:] 
    except requests.RequestException as e:
        yield f"Error: {str(e)}"

def chat_with_model(messages, model="llama3.1"):
    url = "http://localhost:8000/chat"
    payload = {
        "model": model,
        "messages": messages
    }
    try:
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        yield line[6:]  # Remove 'data: ' prefix
    except requests.RequestException as e:
        yield f"Error: {str(e)}"

# Streamlit UI
st.title("Ollama API Tester")

# Embeddings
st.header("Get Embeddings")
embed_text = st.text_input("Enter text for embeddings:")
if st.button("Get Embeddings"):
    embeddings = get_embeddings(embed_text)
    st.write(embeddings)

# Generate
st.header("Generate Response")
gen_model = st.selectbox("Select model for generation", ["llama3.1", "llama3"])
gen_prompt = st.text_area("Enter prompt for generation:")
max_tokens = st.slider("Max tokens", min_value=1, max_value=500, value=100)
if st.button("Generate"):
    response_placeholder = st.empty()
    full_response = ""
    for response in generate_response(gen_prompt, gen_model, max_tokens):
        full_response += response
        response_placeholder.markdown(full_response)

# Chat
st.header("Chat with Model")
chat_model = st.selectbox("Select model for chat", ["llama3.1", "llama3"])
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your message?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for response in chat_with_model(st.session_state.messages, chat_model):
            full_response += response
            response_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})