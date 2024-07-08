import requests
from rich.console import Console
from rich.style import Style
from functools import partial


# Initialize rich for pretty printing
console = Console()
question_style = Style(color="#FF5733", bold=True)  # Red for questions
answer_style = Style(color="#33FF57", bold=True)    # Green for answers
pprint_question = partial(console.print, style=question_style)
pprint_answer = partial(console.print, style=answer_style)

def test_queries():
    url = "http://localhost:8000/ask"
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
        response = requests.post(url, json={"question": query})
        if response.status_code == 200:
            pprint_question(f"Question: {query}")
            pprint_answer(f"Answer: {response.json()['answer']}\n")
        else:
            pprint_question(f"Failed to get response for: {query}")

if __name__ == "__main__":
    test_queries()