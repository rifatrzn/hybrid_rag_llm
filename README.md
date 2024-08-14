# MIE Healthcare Management System

This project is designed to create a Healthcare Management System using FastAPI, Gradio, and NVIDIA AI endpoints. It uses Docker and GitHub Actions for continuous integration and deployment.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features

- FastAPI backend for handling API requests
- Gradio interface for user interaction
- Integration with NVIDIA AI endpoints
- Docker support for containerization
- GitHub Actions for CI/CD

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Docker
- Docker Compose
- NVIDIA GPU and CUDA drivers (if running with GPU support)

### Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/rifatrzn/langchain_rag_llm.git
    cd langchain_rag_llm
    ```

2. **Create a `.env` file with the following content:**

    ```env
    NVIDIA_API_KEY=your_nvidia_api_key
    ```

    Replace `your_nvidia_api_key` with your actual API keys.

### Usage

1. **Build and run the Docker containers:**

    ```sh
    docker-compose up --build
    ```

2. **Access the FastAPI backend:**

    Open your browser and navigate to `http://localhost:8000`

3. **Access the Gradio interface:**

    Open your browser and navigate to `http://localhost:7860`

### Testing

1. **Run tests using Docker:**

    ```sh
    docker run -d --name test-container -p 8000:8000 -p 7860:7860 \
      -e NVIDIA_API_KEY=${NVIDIA_API_KEY} \
      ghcr.io/rifatrzn/rag-server:latest
    ```

2. **Check the logs to verify the container is running:**

    ```sh
    docker logs test-container
    ```

3. **Access the application:**

    - FastAPI endpoint: `http://localhost:8000`
    - Gradio interface: `http://localhost:7860`


# Project Overview

## How the LLM Retriever and HYDE Chain Work

```mermaid
flowchart TD
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef process fill:#d4f1f4,stroke:#05445e,stroke-width:2px;
    classDef data fill:#f9e1e0,stroke:#db6b6b,stroke-width:2px;
    classDef ui fill:#e8f1d4,stroke:#5a8f29,stroke-width:2px;

    subgraph DP[Document Processing]
        A[Fetch Markdown]:::data --> B[Process Markdown]
        B --> C[Split into chunks]
        C --> D[Create embeddings]
        D --> E[Build FAISS index]
        E --> F[Save index]:::data
    end

    subgraph SS[Server Setup]
        G[Load FAISS index]:::data --> H[Initialize Embeddings]
        H --> I[Set up HYDE chain]
        I --> J[Configure FastAPI]:::process
    end

    subgraph QP[Query Processing]
        K[Receive query]:::ui --> L[Generate hypothetical answer]
        L --> M[Retrieve documents]
        M --> N[Generate final answer]:::process
    end

    subgraph UI[User Interface]
        O[Gradio interface]:::ui --> P[Send to FastAPI]
        P --> Q[Display result]:::ui
    end

    F -.-o G
    J --> K
    N --> P

    style DP fill:#f0f8ff,stroke:#333,stroke-width:2px
    style SS fill:#fff0f5,stroke:#333,stroke-width:2px
    style QP fill:#f5fff0,stroke:#333,stroke-width:2px
    style UI fill:#fff5e6,stroke:#333,stroke-width:2px