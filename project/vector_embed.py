import os
import pickle
import shutil
import tarfile
from dotenv import load_dotenv
from functools import partial
from rich.console import Console
from rich.style import Style
from langchain_core.runnables import RunnableLambda, chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set additional environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up console for rich output
console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)

def RPrint(preface="State: "):
    def print_and_return(x, preface=""):
        print(f"{preface}{x}")
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def PPrint(preface="State: "):
    def print_and_return(x, preface=""):
        pprint(preface, x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def create_embeddings(docs_dir: str = './docs', embedding_path: str = "./nomic_embed", chunk_size: int = 500, chunk_overlap: int = 50, batch_size: int = 100):
    # Ensure the directory exists
    os.makedirs(embedding_path, exist_ok=True)
    logger.info(f"Storing embeddings to {embedding_path}")

    # Load Markdown documents
    loader = DirectoryLoader(
        docs_dir, 
        glob="**/*.md", 
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True,
        loader_kwargs={'autodetect_encoding': True}
    )

    try:
        # Load documents
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents")

        # Define headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        # Initialize the MarkdownHeaderTextSplitter
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # Apply the markdown splitter to the text content of each document
        md_header_splits = []
        for doc in docs:
            splits = markdown_splitter.split_text(doc.page_content)
            for split in splits:
                md_header_splits.append(Document(page_content=split.page_content, metadata=doc.metadata))

        logger.info(f"Split documents into {len(md_header_splits)} chunks based on headers")

        # Initialize the RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", ";", ",", " "],
        )

        # Chunk the split documents
        chunked_docs = text_splitter.split_documents(md_header_splits)
        logger.info(f"Further split into {len(chunked_docs)} chunks")

        # Initialize Ollama Embeddings
        logger.info("Initializing embedding model...")
        embeddings = OllamaEmbeddings(base_url="http://ollama:11434", model="nomic-embed-text")
        logger.info("Embedding model initialized.")

        # Create FAISS vector store from chunked documents in batches
        logger.info(f"Creating embeddings for {len(chunked_docs)} chunks in batches of {batch_size}")
        vector_store = None
        for i in tqdm(range(0, len(chunked_docs), batch_size), desc="Processing batches"):
            batch = chunked_docs[i:i+batch_size]
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
                logger.info(f"Processed batch {i//batch_size + 1} of {len(chunked_docs)//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")

        if vector_store:
            # Save the FAISS index to the specified directory
            vector_store.save_local(embedding_path)
            logger.info("All embeddings created and saved successfully.")
        else:
            logger.error("Failed to create any embeddings.")

        return vector_store

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return None
    
if __name__ == "__main__":
    create_embeddings()
