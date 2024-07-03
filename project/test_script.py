import requests

def test_queries():
    url = "http://localhost:8000/ask"
    queries = [
        "What functionalities does the document management system provide?",
        "How does the document management system improve client experience?",
        "What functionalities does the E-Sign module provide?",
        "How does the system support telehealth?"
    ]

    for query in queries:
        response = requests.post(url, json={"question": query})
        if response.status_code == 200:
            print(f"Question: {query}")
            print(f"Answer: {response.json()['answer']}\n")
        else:
            print(f"Failed to get response for: {query}")

if __name__ == "__main__":
    test_queries()