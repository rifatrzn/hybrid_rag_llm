import streamlit as st
import requests
import json

# Set page config at the very start
st.set_page_config(page_title="MIE Healthcare System", layout="wide")

# Add this near the top of the file
SERVER_URL = st.sidebar.text_input("Server URL", value="http://localhost:8001")

def api_request(endpoint, payload):
    url = f"{SERVER_URL}/{endpoint}"
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()  # Return the JSON response
    except requests.RequestException as e:
        st.error(f"Error communicating with the server: {str(e)}")
        return None

def generate_response(prompt, provider, model, max_tokens=100, history=[]):
    payload = {
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "history": history
    }
    return api_request("generate", payload)

# Streamlit UI

# Update Vector Database
with st.sidebar:
    st.header("Settings")
    if st.button("Update Vector DB"):
        with st.spinner("Updating Vector Database..."):
            update_result = api_request("update_vector_db", {})
            if update_result and "Error" not in update_result:
                st.success("Vector Database Updated")
            else:
                st.error("Failed to update Vector Database")
            st.text(update_result)

st.title("MIE Healthcare System")

# Generate Response Section with Chat-like Interface
st.header("Generate Response")

gen_provider = st.selectbox("Select provider", ["ollama", "nvidia"])
gen_model = st.selectbox(
    "Select model", 
    ["llama3.1", "llama3"] if gen_provider.lower() == "ollama" else ["meta/llama-3.1-70b-instruct"]
)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
st.markdown('<div style="background-color: #333333; padding: 20px; border-radius: 8px;">', unsafe_allow_html=True)
for message in st.session_state.chat_history:
    role_color = "#CCCCCC" if message["role"] == "user" else "#AAAAAA"
    st.markdown(f'<div style="color: {role_color};"><strong>{message["role"].capitalize()}:</strong> {message["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input for user prompt
prompt = st.text_area("Type your message here...", height=100)

# Add Send button
if st.button("Send"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Generate response
    with st.spinner("Generating response..."):
        result = generate_response(prompt, gen_provider, gen_model, 100, st.session_state.chat_history)
        if result and "answer" in result:
            st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})
            if "documents" in result and result["documents"]:
                st.subheader("Related Documents:")
                for doc in result["documents"]:
                    st.write(f"Content: {doc['content']}")
                    st.write(f"Source: {doc['metadata']['source']}")
        elif result and "error" in result:
            st.error(result.get("error", "An unexpected error occurred."))

# Add custom CSS styling for dark mode
st.markdown("""
<style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton button {
        background-color: #555555;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
    }
    .stTextInput {
        margin-top: 20px;
    }
    .stTextArea textarea {
        border-radius: 10px;
        padding: 10px;
        background-color: #2C2C2C;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)
