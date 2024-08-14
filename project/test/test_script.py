import requests
from rich.console import Console
from rich.style import Style
from functools import partial

# Initialize rich for pretty printing
console = Console()
question_style = Style(color="#FF5733", bold=True)  # Red for questions
answer_style = Style(color="#33FF57", bold=True)    # Green for answers
error_style = Style(color="#FF3333", bold=True)     # Bold red for errors
pprint_question = partial(console.print, style=question_style)
pprint_answer = partial(console.print, style=answer_style)
pprint_error = partial(console.print, style=error_style)

def test_queries():
    # Ensure the URL matches the running server's port
    url = "http://localhost:8000/chat"  # Updated to use the chat endpoint

    queries = [
        "What functionalities does the system provide for managing patient flow?",
        "How does the system handle potential duplicate patient records?",
        "What functionalities does the document management system provide?",
        "How does the document management system improve client experience?",
        "What are the capabilities of the printing and print definitions feature?",
        "How does the system facilitate the access and manipulation of diagnostic radiological images?",
        "What capabilities does the E-Chart module offer?",
        "What functionalities does the E-Sign module provide?",
        "How does the system handle injury care and reporting?",
        "What functionalities does the system provide for order management?"
    ]

    for query in queries:
        try:
            response = requests.post(url, json={
                "model": "llama3.1",
                "messages": [{"role": "user", "content": query}]
            })
            if response.status_code == 200:
                pprint_question(f"Question: {query}")
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix
                            if data.startswith('DOCUMENT_INFO:'):
                                doc_info = eval(data[15:])  # Parse document info
                                pprint_answer("Related Documents:")
                                for doc in doc_info:
                                    pprint_answer(f"Content: {doc['content'][:100]}...")
                                    pprint_answer(f"Metadata: {doc['metadata']}")
                            else:
                                full_response += data
                pprint_answer(f"Answer: {full_response}\n")
            else:
                pprint_error(f"Failed to get response for: {query} with status code: {response.status_code}")
                pprint_error(f"Response: {response.text}")
        except requests.exceptions.RequestException as e:
            pprint_error(f"An error occurred while making a request for: {query}")
            pprint_error(f"Error: {str(e)}\n")

if __name__ == "__main__":
    test_queries()