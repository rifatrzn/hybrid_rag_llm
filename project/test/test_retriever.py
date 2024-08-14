from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retrieve_documents(vector_store_path: str, query: str, top_k: int = 5):
    try:
        # Load the FAISS index from the specified directory
        embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
        vector = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        logger.info(f"Performing similarity search for query: '{query}'")
        # Perform a similarity search
        results = vector.similarity_search(query, k=top_k)
        
        return results

    except Exception as e:
        logger.error(f"Error occurred during retrieval: {str(e)}")
        return []

def main():
    embedding_path = "./nomic_embed"
    question = "What is the best type of e-chart module for healthcare system?"
    
    # Retrieve documents
    docs = retrieve_documents(embedding_path, question)
    
    logger.info(f"Number of relevant documents retrieved: {len(docs)}")
    
    # Print the content of the retrieved documents
    for i, doc in enumerate(docs):
        logger.info(f"\nResult {i+1}:")
        logger.info(f"Content: {doc.page_content[:500]}")  # Print the first 500 characters
        logger.info(f"Metadata: {doc.metadata}\n")

if __name__ == "__main__":
    main()

    
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

def main():
    embedding_path = "./nomic_embed"
    question = "What is the best type of chart for healthcare system?"
    
    # Load the vector store
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
    vectorstore = FAISS.load_local(embedding_path, embeddings, allow_dangerous_deserialization=True)
    
    # Create the Ollama language model
    ollama = Ollama(base_url="http://localhost:11434", model="llama3.1")
    
    # Create the RetrievalQA chain
    qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
    
    # Invoke the chain with the question
    res = qachain.invoke({"query": question})
    
    logger.info("Model's answer:")
    logger.info(res['result'])

if __name__ == "__main__":
    main()