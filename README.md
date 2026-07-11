# 🧠 Explainable Multi-Document RAG System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-4B0082)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Persistent-00C853)](https://www.trychroma.com/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-4285F4)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<img src="https://img.shields.io/badge/Status-🚧%20Active%20Development-orange?style=for-the-badge"/>

### Production-Inspired Explainable Retrieval-Augmented Generation (RAG) System

*A modular, end-to-end AI application demonstrating modern Retrieval-Augmented Generation, semantic search, explainable responses, and production-ready software engineering practices.*

⭐ **If you find this project interesting, consider giving it a star!**

</div>

---

# 🚧 Project Status

> **Status:** Active Development

This repository is actively being developed using an **incremental feature-based Git workflow** to simulate a real-world AI software engineering lifecycle.

Rather than uploading the final project in a single commit, every major feature is implemented, tested, documented, and committed independently, making the repository reflect how production software is actually developed.

### Current Progress

- ✅ Core RAG Pipeline Completed
- ✅ Explainable Responses with Citations
- ✅ Multi-document Retrieval
- ✅ Streamlit Dashboard
- 🚧 FastAPI Backend (Currently in Development)
- ⏳ Production Deployment
- ⏳ Enterprise Features

---

# 🌟 Project Overview

Large Language Models (LLMs) are powerful, but they often generate answers that are **not grounded in factual evidence**. This project addresses that challenge by implementing an **Explainable Multi-Document Retrieval-Augmented Generation (RAG) System** capable of retrieving relevant information from multiple PDF documents before generating responses.

Instead of relying solely on the model's internal knowledge, the system first performs semantic retrieval over a persistent vector database, selects the most relevant document chunks, constructs a grounded prompt, and finally generates an evidence-backed answer using **Gemini 2.5 Flash**.

Unlike a basic RAG implementation, this project is designed with **software engineering best practices** in mind, emphasizing modularity, explainability, maintainability, and scalability.

---

# 🎯 Project Objectives

This project aims to demonstrate practical AI engineering concepts including:

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Explainable AI
- Multi-document Question Answering
- Vector Databases
- Prompt Engineering
- Production-Oriented Architecture
- Modular Python Development
- End-to-End AI Application Development

---

# ✨ Key Features

## ✅ Currently Implemented

| Feature | Description |
|----------|-------------|
| 📄 Multi-document PDF Loading | Load and process multiple PDF documents simultaneously |
| ✂️ Intelligent Text Chunking | Split documents into semantic chunks while preserving metadata |
| 🧠 Embedding Generation | Generate vector embeddings for semantic search |
| 🗄️ Persistent ChromaDB | Store document embeddings efficiently for reuse |
| 🔍 Semantic Retrieval | Retrieve relevant document chunks based on user queries |
| 🤖 RAG Generation Pipeline | Generate grounded answers using retrieved context |
| 📚 Source Citations | Provide transparent evidence supporting generated responses |
| 📊 Streamlit Dashboard | Interactive interface to inspect the RAG pipeline |
| 📁 Modular Architecture | Clean, maintainable project organization |
| 📑 Multi-document Support | Query information across multiple documents |
| 🎯 Similarity Filtering | Reject weak retrieval matches to reduce hallucinations |

---

## 🚧 Currently Under Development

The project is now evolving from a functional prototype into a production-ready AI application.

### Immediate Goals

- FastAPI Backend
- REST API
- Swagger Documentation
- Streamlit–FastAPI Integration

---

## 🛣️ Production Roadmap

| Phase | Feature | Status |
|---------|:------:|:------:|
| Project Setup | ✅ | Completed |
| PDF Loading | ✅ | Completed |
| Document Chunking | ✅ | Completed |
| Embeddings | ✅ | Completed |
| Vector Database | ✅ | Completed |
| Semantic Retrieval | ✅ | Completed |
| End-to-End RAG Pipeline | ✅ | Completed |
| Source Citations | ✅ | Completed |
| Streamlit Dashboard | ✅ | Completed |
| Multi-document Retrieval | ✅ | Completed |
| Similarity Thresholding | ✅ | Completed |
| FastAPI Backend | 🚧 | In Progress |
| Streamlit ↔ FastAPI Integration | ⏳ | Planned |
| Multilingual Support | ⏳ | Planned |
| Contradiction Detection API | ⏳ | Planned |
| Confidence Scoring | ⏳ | Planned |
| Cross-Encoder Re-ranking | ⏳ | Planned |
| Retrieval Evaluation | ⏳ | Planned |
| Structured Logging | ⏳ | Planned |
| Embedding Caching | ⏳ | Planned |
| Docker Support | ⏳ | Planned |
| Cloud Deployment | ⏳ | Planned |
| Testing Suite | ⏳ | Planned |
| Production Documentation | ⏳ | Planned |

---

# 📈 Development Progress

| Stage | Progress |
|---------|---------|
| Core RAG Pipeline | ██████████ 100% |
| Explainability | ██████████ 100% |
| Multi-document Retrieval | ██████████ 100% |
| User Interface | ██████████ 100% |
| Backend API | ███░░░░░░░ 30% |
| Production Features | ██░░░░░░░░ 20% |
| Deployment | ░░░░░░░░░░ 0% |

---

# 💡 Why This Project?

Most beginner RAG projects demonstrate only the minimum workflow of loading documents, retrieving chunks, and generating answers.

This repository goes beyond that by focusing on:

- Production-inspired architecture
- Explainable AI with source citations
- Modular and maintainable code
- Real-world software engineering workflow
- Incremental feature-based Git history
- Scalable project organization
- Future production deployment

The long-term vision is to evolve this repository into a complete production-ready AI service with REST APIs, multilingual support, evaluation pipelines, deployment, testing, Docker, confidence scoring, contradiction detection, and advanced retrieval techniques.

---

# 📌 Repository Note

> This repository is intentionally developed through **small, feature-based commits** rather than a single upload. Each feature is implemented, tested, documented, and committed independently to reflect a realistic professional software development workflow.
>
> ````markdown
---

# 🏛️ System Architecture

The application follows a modular Retrieval-Augmented Generation (RAG) architecture, where each component has a single responsibility. This design improves maintainability, scalability, and makes it easier to replace individual components without affecting the entire system.

```mermaid
flowchart TD

    User([User Query])

    User --> UI

    subgraph Frontend
        UI[Streamlit Dashboard]
    end

    UI --> Retriever

    subgraph RAG Pipeline

        Loader[PDF Loader]

        Chunker[Text Splitter]

        Embeddings[Embedding Model]

        VectorDB[(ChromaDB)]

        Retriever[Semantic Retriever]

        Prompt[Prompt Builder]

        LLM[Gemini 2.5 Flash]

        Citation[Citation Formatter]

    end

    Loader --> Chunker

    Chunker --> Embeddings

    Embeddings --> VectorDB

    VectorDB --> Retriever

    Retriever --> Prompt

    Prompt --> LLM

    LLM --> Citation

    Citation --> Response[Grounded Response]

````

> **Current Architecture**
>
> The Streamlit dashboard directly interacts with the RAG pipeline.
>
> The next milestone is to introduce a FastAPI backend, allowing the frontend and backend to operate independently through REST APIs.

---

# 🔄 End-to-End Workflow

The following workflow illustrates how user queries are processed from raw PDF documents to explainable answers.

```text
                 PDF Documents
                       │
                       ▼
               PDF Loading Module
                       │
                       ▼
             Text Chunking + Metadata
                       │
                       ▼
            Embedding Generation
                       │
                       ▼
           Persistent ChromaDB Index
                       │
──────────────────────────────────────────────────────
                  User submits Query
──────────────────────────────────────────────────────
                       │
                       ▼
            Semantic Similarity Search
                       │
                       ▼
             Retrieve Top-K Chunks
                       │
                       ▼
             Prompt Construction
                       │
                       ▼
              Gemini 2.5 Flash
                       │
                       ▼
            Citation Formatting
                       │
                       ▼
        Explainable Grounded Response
```

---

# 📂 Project Structure

```text
.
├── app.py                         # Streamlit dashboard
├── README.md
├── requirements.txt
├── .env
│
├── assets/                        # Images, screenshots, diagrams
│
├── chroma_db/                     # Persistent vector database
│
├── data/                          # Source PDF documents
│
├── scripts/
│   └── check.py
│
└── src/
    ├── main.py
    │
    ├── loaders/
    │     └── pdf_loader.py
    │
    ├── processors/
    │     └── text_splitter.py
    │
    ├── embeddings/
    │     └── vector_store.py
    │
    ├── retrievers/
    │     └── semantic_retriever.py
    │
    ├── prompts/
    │     └── prompt_template.py
    │
    ├── llm/
    │     └── generator.py
    │
    └── citations/
          └── formatter.py
```

---

# 📖 Module Responsibilities

| Module                  | Responsibility                         |
| ----------------------- | -------------------------------------- |
| `app.py`                | Streamlit dashboard and user interface |
| `main.py`               | Executes the complete RAG workflow     |
| `pdf_loader.py`         | Reads PDF documents                    |
| `text_splitter.py`      | Splits documents into semantic chunks  |
| `vector_store.py`       | Creates and manages ChromaDB           |
| `semantic_retriever.py` | Retrieves relevant document chunks     |
| `prompt_template.py`    | Builds grounded prompts                |
| `generator.py`          | Generates answers using Gemini         |
| `formatter.py`          | Adds citations to generated responses  |

---

# 🛠️ Technology Stack

| Layer                | Technology            |
| -------------------- | --------------------- |
| Programming Language | Python 3.10+          |
| LLM                  | Gemini 2.5 Flash      |
| AI Framework         | LangChain             |
| Vector Database      | ChromaDB              |
| Embedding Model      | Sentence Transformers |
| UI Framework         | Streamlit             |
| PDF Processing       | PyPDF                 |
| Configuration        | python-dotenv         |
| Version Control      | Git & GitHub          |

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git

cd <repository-name>
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

Activate it

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure Environment Variables

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

---

## 5️⃣ Add Documents

Place your PDF files inside

```text
data/
```

Example

```text
data/
│
├── rag.pdf
├── transformers.pdf
├── llm.pdf
```

---

## 6️⃣ Run the Application

Launch the Streamlit dashboard

```bash
streamlit run app.py
```

---

# ▶️ Running the CLI Version

To execute the pipeline directly without the dashboard

```bash
python -m src.main
```

---

# 📸 Demo

The following visuals will be added as development progresses.

| Preview                 | Status         |
| ----------------------- | -------------- |
| Architecture Diagram    | 🚧 Coming Soon |
| Streamlit Dashboard     | 🚧 Coming Soon |
| Retrieval Visualization | 🚧 Coming Soon |
| Demo GIF                | 🚧 Coming Soon |
| API Documentation       | 🚧 Coming Soon |

---

# 🔑 Environment Variables

| Variable         | Description                               |
| ---------------- | ----------------------------------------- |
| `GOOGLE_API_KEY` | Gemini API Key used for answer generation |

---

# 📦 Current Project Outputs

The current application provides

* ✅ Grounded answers
* ✅ Retrieved supporting chunks
* ✅ Source citations
* ✅ Multi-document retrieval
* ✅ Interactive dashboard
* ✅ Persistent vector database

Future releases will additionally include

* FastAPI endpoints
* JSON APIs
* Docker containers
* Deployment
* Confidence scoring
* Multilingual support
* Cross-encoder reranking
* Evaluation pipeline



---

# 💬 Example Usage

## Example Query

```text
What is Retrieval-Augmented Generation (RAG)?
```

---

## Retrieved Context

```
Document: rag.pdf

Chunk 1:
Retrieval-Augmented Generation (RAG) improves the factual accuracy
of LLMs by retrieving relevant information from external knowledge
sources before generating responses.

Chunk 2:
The retrieved context is incorporated into the prompt, allowing
the language model to generate grounded and explainable answers.
```

---

## Generated Response

> Retrieval-Augmented Generation (RAG) is an AI architecture that combines information retrieval with Large Language Models (LLMs). Before generating an answer, the system retrieves relevant information from external documents and incorporates that context into the prompt. This grounding process significantly reduces hallucinations while improving factual accuracy and explainability.

---

## Source Citations

```
Sources

• rag.pdf (Page 3)
• transformers.pdf (Page 12)
```

---

# 🖥️ Streamlit Dashboard

The interactive Streamlit dashboard provides complete visibility into each stage of the RAG pipeline, making the retrieval and generation process transparent.

## Current Dashboard Features

- 📄 Ask questions across multiple documents
- 🔍 Inspect retrieved document chunks
- 📚 View supporting citations
- 🧠 Analyze prompt construction
- 📊 Display retrieval metadata
- ⚡ Measure response latency
- 📈 Monitor retrieval quality

---

## Planned Dashboard Enhancements

- Confidence visualization
- Interactive retrieval score charts
- Document similarity heatmaps
- Retrieval analytics
- Query history
- Conversation memory
- Source comparison
- API monitoring

---

# 🧠 Engineering Decisions

This project intentionally prioritizes **software engineering best practices** alongside AI functionality.

Every major design decision was made to improve modularity, scalability, maintainability, and explainability.

---

## Why Retrieval-Augmented Generation?

Traditional Large Language Models rely only on their pre-trained knowledge.

As a result, they may:

- Hallucinate facts
- Produce outdated information
- Lack transparency
- Fail on domain-specific documents

Retrieval-Augmented Generation addresses these limitations by grounding every response using relevant external knowledge retrieved at runtime.

---

## Why ChromaDB?

Several vector databases were evaluated before selecting ChromaDB.

Advantages include:

- Lightweight
- Persistent storage
- Fast semantic search
- Excellent LangChain integration
- Easy local development
- Production-friendly migration path

---

## Why Gemini 2.5 Flash?

The project uses Gemini 2.5 Flash because it provides:

- Excellent reasoning capability
- Fast inference
- Cost-effective API usage
- Strong instruction following
- Reliable grounded generation

This makes it suitable for interactive Retrieval-Augmented Generation applications.

---

## Why a Modular Architecture?

Instead of implementing everything inside a single script, the project separates responsibilities into independent modules.

Benefits include:

- Easier testing
- Better maintainability
- Cleaner codebase
- Simpler debugging
- Improved scalability
- Production readiness

Each module has a single responsibility, making future enhancements significantly easier.

---

## Why Feature-Based Git Commits?

This repository is intentionally developed through small, incremental commits.

Rather than uploading the final project all at once, each feature is:

- Designed
- Implemented
- Tested
- Documented
- Committed independently

This mirrors professional software development practices and makes the evolution of the project easy to follow.

---

# ⚡ Performance Considerations

Current optimizations include:

- Persistent vector storage
- Reusable embeddings
- Semantic similarity filtering
- Modular prompt construction
- Efficient document chunking

Future releases will introduce:

- Embedding caching
- API caching
- Cross-encoder reranking
- Hybrid retrieval
- Asynchronous inference
- Response streaming

---

# 📊 Current Project Maturity

| Capability | Status |
|------------|:------:|
| PDF Processing | ✅ |
| Semantic Retrieval | ✅ |
| Explainable Responses | ✅ |
| Source Citations | ✅ |
| Multi-document Support | ✅ |
| Interactive Dashboard | ✅ |
| FastAPI Backend | 🚧 |
| REST API | 🚧 |
| Deployment | ⏳ |
| Docker | ⏳ |
| Testing | ⏳ |
| Production Monitoring | ⏳ |

---

# ⚠️ Current Limitations

Although the core RAG pipeline is fully functional, several production-oriented capabilities are still under development.

Current limitations include:

- Semantic retrieval only
- No cross-encoder reranking
- No multilingual support
- No REST API
- No conversation memory
- No metadata filtering
- No automated evaluation framework
- No confidence scoring
- No deployment pipeline

These limitations are already included in the project roadmap and will be implemented incrementally.

---

# 🎓 Learning Outcomes

This project demonstrates practical experience with:

## Artificial Intelligence

- Retrieval-Augmented Generation (RAG)
- Prompt Engineering
- Explainable AI
- Semantic Search
- Embedding Models
- Vector Databases

---

## Software Engineering

- Modular Python Development
- Clean Architecture
- Separation of Concerns
- Incremental Git Workflow
- Documentation
- Maintainable Code Design

---

## Backend Development

- Data Processing
- Pipeline Design
- API-Oriented Architecture
- Configuration Management
- Persistent Storage

---

## Future Skills Covered

The remaining roadmap will additionally demonstrate experience with:

- FastAPI
- REST APIs
- Docker
- Deployment
- Cross-Encoder Models
- Evaluation Pipelines
- Confidence Scoring
- Logging
- Production Monitoring

---

# 🌟 Why This Project Stands Out

Unlike many introductory RAG implementations, this repository focuses on building a production-inspired AI system rather than a simple proof of concept.

Key differentiators include:

- Explainable responses with citations
- Modular software architecture
- Persistent vector database
- Multi-document retrieval
- Feature-based Git history
- Production-focused roadmap
- Professional documentation
- Continuous incremental development

The long-term objective is to evolve this repository into a deployable, production-ready AI service that demonstrates modern AI engineering practices from data ingestion to scalable deployment.




# 🚀 Production Roadmap

The current implementation provides a fully functional explainable RAG pipeline. The next development phases focus on transforming it into a production-ready AI service.

| Phase | Feature | Priority | Status |
|---------|----------|:--------:|:------:|
| Phase 1 | FastAPI Backend | 🔥 High | 🚧 In Progress |
| Phase 2 | Streamlit → FastAPI Integration | 🔥 High | ⏳ Planned |
| Phase 3 | Multilingual Support | 🔥 High | ⏳ Planned |
| Phase 4 | Document Contradiction Detection | ⭐ Medium | ⏳ Planned |
| Phase 5 | Confidence Scoring | ⭐ Medium | ⏳ Planned |
| Phase 6 | Cross-Encoder Re-ranking | ⭐ Medium | ⏳ Planned |
| Phase 7 | Retrieval Evaluation Pipeline | ⭐ Medium | ⏳ Planned |
| Phase 8 | Structured Logging | ⭐ Medium | ⏳ Planned |
| Phase 9 | Embedding & API Caching | ⭐ Medium | ⏳ Planned |
| Phase 10 | Docker Support | 🚀 High | ⏳ Planned |
| Phase 11 | Cloud Deployment | 🚀 High | ⏳ Planned |
| Phase 12 | Testing & CI/CD | 🚀 High | ⏳ Planned |

---

# 🔮 Future Enhancements

Beyond the production roadmap, several advanced AI engineering features are planned to further improve retrieval quality, scalability, and user experience.

### Retrieval Improvements

- Hybrid Search (BM25 + Semantic Retrieval)
- Metadata-Based Filtering
- Query Expansion
- Context Compression
- Adaptive Chunk Selection
- Dynamic Top-K Retrieval

---

### LLM Enhancements

- Conversational Memory
- Multi-turn Question Answering
- Tool Calling
- Self-RAG
- Reflection-Based Answer Verification
- Agentic Workflows

---

### Explainability

- Confidence Scores
- Source Reliability Ranking
- Retrieval Score Visualization
- Evidence Highlighting
- Contradiction Analysis

---

### Production Features

- Docker
- Kubernetes Deployment
- Redis Caching
- Monitoring Dashboards
- Health Check Endpoints
- Authentication
- Rate Limiting

---

### Evaluation

- Retrieval Precision
- Retrieval Recall
- Top-K Accuracy
- Response Faithfulness
- Latency Benchmarks
- Hallucination Evaluation

---

# 🤝 Contributing

Contributions are always welcome.

Whether you're fixing bugs, improving documentation, optimizing retrieval quality, or proposing new AI features, your contributions are appreciated.

## How to Contribute

1. Fork the repository

2. Create a feature branch

```bash
git checkout -b feature/my-feature
```

3. Commit your changes

```bash
git commit -m "feat: add awesome feature"
```

4. Push your branch

```bash
git push origin feature/my-feature
```

5. Open a Pull Request

Please ensure your code follows the existing project structure and includes appropriate documentation where necessary.

---

# 🛠️ Development Philosophy

This repository follows several engineering principles.

### Clean Architecture

Every component has a single responsibility.

---

### Modular Design

Each stage of the RAG pipeline is implemented independently, making the system easy to extend and maintain.

---

### Explainability First

Every generated response should be grounded in retrieved evidence whenever possible.

---

### Incremental Development

Features are added through small, focused Git commits that document the evolution of the project over time.

---

### Production Mindset

Although this project started as a learning exercise, every new feature is implemented with scalability, maintainability, and deployment in mind.

---

# 📊 Repository Statistics

Current project highlights

- ✅ 12+ Feature-Based Commits
- ✅ Modular Codebase
- ✅ Multi-Document Retrieval
- ✅ Persistent Vector Database
- ✅ Explainable Responses
- ✅ Interactive Dashboard
- 🚧 Production API
- 🚧 Enterprise Features

---

# 🙏 Acknowledgements

This project builds upon several outstanding open-source technologies.

Special thanks to:

- Python
- LangChain
- Google Gemini
- ChromaDB
- Streamlit
- PyPDF
- Sentence Transformers
- Open Source AI Community

Without these technologies, building modern Retrieval-Augmented Generation systems would be significantly more challenging.

---

# 📚 References

Some of the concepts implemented in this repository are inspired by research and engineering practices from:

- Retrieval-Augmented Generation (RAG)
- Vector Databases
- Semantic Search
- Prompt Engineering
- Explainable AI
- Modern LLM Application Development

---

# 📜 License

This project is licensed under the MIT License.

Feel free to use, modify, and distribute this project in accordance with the license terms.

---

# 👨‍💻 Author

**Shreyash Deshmukh**

Final Year Computer Engineering Student

Passionate about AI Engineering, Retrieval-Augmented Generation, Large Language Models, Backend Systems, and Production-Ready AI Applications.

---

<div align="center">

## ⭐ Support the Project

If you found this repository helpful or interesting, consider giving it a ⭐ on GitHub.

It helps increase the visibility of the project and motivates further development.

---

### 🚧 This project is actively evolving toward a production-ready AI system.

**New features, architectural improvements, and documentation updates are added regularly.**

Thanks for visiting the repository! 🚀

</div>
````


