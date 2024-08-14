# test_embedding_endpoint.py
import requests
import json

def test_embedding():
    url = "http://localhost:8000/embeddings"  # Adjust this URL according to your setup
    payload = {
        "text": "This is a test sentence for embeddings."
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        print("Embeddings response:", response.json())
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    test_embedding()