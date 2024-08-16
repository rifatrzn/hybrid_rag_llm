# MIE Healthcare Enterprise

🏥 A powerful and intuitive healthcare information system powered by advanced language models and vector databases.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [System Diagram](#system-diagram)
5. [Technologies Used](#technologies-used)
6. [Setup and Installation](#setup-and-installation)
7. [Usage](#usage)
8. [Deployment](#deployment)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview

MIE Healthcare Enterprise is a sophisticated healthcare information system that leverages cutting-edge AI technologies to provide intelligent document management, question answering, and information retrieval capabilities. The system uses a combination of large language models (LLMs) and vector databases to process and analyze healthcare-related documents, offering a user-friendly interface for healthcare professionals to interact with the system.

## Features

1. 🤖 AI-powered question answering system
2. 📚 Intelligent document retrieval and management
3. 🔍 Advanced search capabilities using vector embeddings
4. 🔄 Real-time vector database updates
5. 🖥️ User-friendly Gradio interface
6. 🚀 Scalable architecture using Docker and Docker Compose
7. 🧠 Support for multiple LLM providers (Ollama and NVIDIA)

## Architecture

The MIE Healthcare Enterprise system consists of several interconnected components:

1. **FastAPI Backend (`app_server.py`)**: Handles API requests, manages the vector database, and orchestrates the LLM interactions.
2. **Gradio UI (`gradio_ui.py`)**: Provides a user-friendly web interface for interacting with the system.
3. **Vector Database Update Script (`update_vector_db.sh`)**: Keeps the vector database up-to-date with the latest document changes.
4. **Vector Embedding Script (`vector_embed.py`)**: Processes documents and creates vector embeddings for efficient search and retrieval.
5. **Docker Compose Configuration (`docker-compose.yaml`)**: Defines the multi-container Docker application for easy deployment and scaling.

## System Diagram

Below is a detailed diagram illustrating the architecture and data flow of the MIE Healthcare Enterprise system:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#6366f1', 'primaryTextColor': '#fff', 'primaryBorderColor': '#4f46e5', 'lineColor': '#64748b', 'secondaryColor': '#f1f5f9', 'tertiaryColor': '#e2e8f0'}}}%%
flowchart TD
    classDef default fill:#f1f5f9,stroke:#6366f1,stroke-width:2px,color:#334155;
    classDef api fill:#818cf8,stroke:#4f46e5,stroke-width:2px,color:#fff;
    classDef llm fill:#22c55e,stroke:#16a34a,stroke-width:2px,color:#fff;
    classDef db fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff;
    classDef ui fill:#e879f9,stroke:#d946ef,stroke-width:2px,color:#fff;
    classDef note fill:#fef3c7,stroke:#f59e0b,stroke-width:2px,color:#92400e;
    classDef update fill:#fca5a5,stroke:#ef4444,stroke-width:2px,color:#fff;

    A["👤 User"]
    B["🖥️ Gradio UI\n(Port 7860)"]:::ui
    C["🚀 FastAPI Backend\n(Port 8001)"]:::api

    subgraph Frontend ["Gradio Frontend"]
        D["💬 Chat Interface\nUser inputs messages"]:::ui
        E["⚙️ Settings Panel\nSelect provider & model"]:::ui
        F["🔄 Update DB Button\nTrigger vector DB update"]:::ui
    end

    subgraph Backend ["FastAPI Backend"]
        G["📊 /embeddings\nGenerate text embeddings"]:::api
        H["💬 /generate\nProcess chat & generate responses"]:::api
        I["🔄 /update_vector_db\nUpdate vector database"]:::api
        J["🏠 /\nRoot endpoint"]:::api
        K["❤️ /health\nHealth check endpoint"]:::api
    end

    subgraph LLM ["LLM Processing"]
        L["🔍 HYDE Query\nGenerate hypothetical document"]:::llm
        M["🗃️ Retrieval\nFetch relevant docs from FAISS"]:::db
        N["❓ QA Chain\nGenerate final answer"]:::llm
        O{"🔀 LLM Selection\nChoose LLM based on settings"}:::llm
        P["🧠 NVIDIA LLM\nmeta/llama-3.1-70b-instruct"]:::llm
        Q["🦙 Ollama LLM\nllama3.1 or llama3"]:::llm
    end

    R["📚 FAISS Index\nStore document embeddings"]:::db
    S["🔤 Ollama Embeddings\nGenerate embeddings"]:::llm

    subgraph UpdateProcess ["Update Process"]
        T["🔄 update_vector_db.sh\nBash script"]:::update
        U["📥 Git Clone/Pull\nFetch latest docs"]:::update
        V["🔨 vector_embed.py\nProcess markdown & update DB"]:::update
    end

    A -->|"1. Interact with UI"| B
    B -->|"2. Send API requests"| C
    C -->|"8. Send response"| B
    B -->|"9. Display response"| A

    D -->|"3. Send message"| H
    E -->|"3. Set LLM preferences"| H
    F -->|"3. Trigger update"| I

    H -->|"4. Process request"| LLM
    I -->|"Trigger update"| T
    G -->|"Generate"| S

    L -->|"5. Create HYDE query"| M
    M -->|"6. Fetch relevant docs"| N
    N -->|"7. Generate answer"| O
    O -->|"If NVIDIA selected"| P
    O -->|"If Ollama selected"| Q
    S -->|"Store embeddings"| R
    M -->|"Retrieve docs"| R

    T -->|"Run"| U
    U -->|"Update docs"| V
    V -->|"Refresh"| R

    note1["📝 Note: The user interacts with the Gradio UI,\nwhich sends requests to the FastAPI backend.\nThe backend processes these requests using\nthe selected LLM and returns the responses."]:::note
    note2["📝 Note: The update process can be triggered\nmanually or automatically. It fetches the latest\ndocumentation, processes it, and updates the\nFAISS index to keep the knowledge base current."]:::note

    style Frontend fill:#f0e6fa,stroke:#d8b4fe,stroke-width:4px
    style Backend fill:#e0f2fe,stroke:#7dd3fc,stroke-width:4px
    style LLM fill:#bbf7d0,stroke:#22c55e,stroke-width:4px
    style UpdateProcess fill:#fee2e2,stroke:#fca5a5,stroke-width:4px
    linkStyle default stroke:#64748b,stroke-width:2px;
```

### Diagram Explanation

1. **User Interface (👤)**:
   - Users interact with the system through a Gradio Web UI, which provides an intuitive interface for submitting queries and receiving responses.
   - The UI communicates with the FastAPI Backend to process user requests.

2. **Backend Services (🖥️)**:
   - The FastAPI Backend receives user queries and routes them to the appropriate services.
   - For each query, the system may:
     a. Generate embeddings using Ollama Embeddings.
     b. Query the FAISS Vector Store to find relevant documents.
     c. Generate a response using one of two LLM providers: Ollama LLM or NVIDIA AI Endpoints.

3. **Data Management (📊)**:
   - The Update Vector DB Script periodically fetches the latest documents from a GitHub repository.
   - The Vector Embedding Script processes these documents and updates the FAISS Vector Store with new embeddings.

4. **Document Storage (📚)**:
   - Documents are stored in a local file system after being fetched from the GitHub repository.

This architecture ensures efficient processing of user queries, real-time updates to the knowledge base, and flexibility in choosing the most appropriate LLM for each task.


## Technologies Used

1. **Language Models**:
   - 🦙 Ollama: Used for embedding generation and as an LLM provider
   - 🖥️ NVIDIA AI Endpoints: Provides access to powerful NVIDIA-hosted language models

2. **Vector Database**:
   - FAISS: Efficient similarity search and clustering of dense vectors

3. **Frameworks and Libraries**:
   - FastAPI: High-performance web framework for building APIs
   - Gradio: Quickly create UIs for machine learning models
   - LangChain: Framework for developing applications powered by language models
   - Pydantic: Data validation and settings management using Python type annotations

4. **Containerization and Orchestration**:
   - Docker: Containerization platform
   - Docker Compose: Tool for defining and running multi-container Docker applications

5. **Other Tools**:
   - Rich: Library for rich text and beautiful formatting in the terminal
   - Tqdm: Fast, extensible progress bar for Python and CLI

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/rifatrzn/langchain_rag_llm.git
   cd mie-healthcare-enterprise
   ```

2. Create a `.env` file in the project root and add your NVIDIA API key:
   ```
   NVIDIA_API_KEY=your_nvidia_api_key_here
   ```

3. Build and start the Docker containers:
   ```
   docker-compose up --build
   ```

4. The system will be available at:
   - FastAPI Backend: http://localhost:8001
   - Gradio UI: http://localhost:7860

## Usage

1. Open the Gradio UI in your web browser (http://localhost:7860).
2. Use the chat interface to ask questions about healthcare documents.
3. Adjust the settings to change the LLM provider, model, and other parameters.
4. Click the "Update Vector Database" button to refresh the document embeddings.

## Deployment

For host deployment, consider the following steps:

1. **Server Requirements**:
   - Ensure the host has Docker and Docker Compose installed.
   - For optimal performance, use a machine with a NVIDIA GPU and proper CUDA setup.

2. **Environment Variables**:
   - Set up the necessary environment variables, especially the `NVIDIA_API_KEY`.

3. **Network Configuration**:
   - Configure the firewall to allow incoming traffic on ports 8001 (FastAPI) and 7860 (Gradio UI).
   - Set up a reverse proxy (e.g., Nginx) to handle HTTPS and domain routing.

4. **Scaling**:
   - Adjust the `docker-compose.yaml` file to scale services as needed.
   - Consider using Docker Swarm or Kubernetes for more advanced orchestration.

5. **Monitoring and Logging**:
   - Set up monitoring tools (e.g., Prometheus, Grafana) to track system performance.
   - Configure centralized logging (e.g., ELK stack) for easier troubleshooting.

6. **Backup and Recovery**:
   - Implement regular backups of the vector database and any persistent data.
   - Create a disaster recovery plan to ensure system availability.

7. **Updates and Maintenance**:
   - Regularly update the document repository and vector database.
   - Keep the Docker images and dependencies up-to-date.

## Contributing

We welcome contributions to the MIE Healthcare Enterprise project! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with clear, descriptive messages
4. Push your changes to your fork
5. Create a pull request with a detailed description of your changes

## License

This project is licensed under the [MIT License](LICENSE).

---

For more information or support, please open an issue on the GitHub repository or contact the project maintainers.
