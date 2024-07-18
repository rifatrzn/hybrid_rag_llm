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

graph TD
    classDef box fill:#f9f,stroke:#333,stroke-width:4px;
    classDef process fill:#0f0,stroke:#333,stroke-width:2px;
    classDef subgraphStyle fill:#e6e6e6,stroke:#333,stroke-width:2px;

    subgraph "Document Processing"
        direction TB
        A[Fetch Markdown from GitHub] -->|Clone repo| B[Process Markdown]
        B --> C[Split into chunks]
        C --> D[Create embeddings]
        D --> E[Build FAISS index]
        E --> F[Save compressed index]
        class A,B,C,D,E,F box
        class "Document Processing" subgraphStyle
    end

    subgraph "Server Setup"
        direction TB
        G[Load FAISS index] --> H[Initialize NVIDIA Embeddings]
        H --> I[Set up HYDE chain]
        I --> J[Configure FastAPI server]
        class G,H,I,J process
        class "Server Setup" subgraphStyle
    end

    subgraph "Query Processing"
        direction TB
        K[Receive user query] --> L[Generate hypothetical answer using HYDE]
        L --> M[Retrieve relevant documents from FAISS index]
        M --> N[Generate final answer using LLM]
        class K,L,M,N process
        class "Query Processing" subgraphStyle
    end

    subgraph "User Interface"
        direction TB
        O[Gradio interface] --> P[Send query to FastAPI]
        P --> Q[Display answer and sources]
        class O,P,Q box
        class "User Interface" subgraphStyle
    end

    F -.->|Load| G
    J --> K
    N --> P
