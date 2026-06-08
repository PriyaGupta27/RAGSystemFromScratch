## RAG System from Basic

### Overview

This project includes a basic implementation of a Retrieval-Augmented Generation (RAG) system built with Streamlit. RAG combines the power of large language models with external knowledge retrieval to provide more accurate and contextually relevant answers. The system allows users to load documents from URLs, process them into chunks, store them in a vector database, and query them using natural language. (Folder: BasicRagSystem)

### Core Concept

Retrieval-Augmented Generation (RAG) addresses the limitations of large language models by grounding their responses in external knowledge sources. Instead of relying solely on the model's training data, RAG systems:

1. **Retrieve** relevant information from a knowledge base
2. **Augment** the query with this retrieved context  
3. **Generate** answers based on the combined information

This approach reduces hallucinations, improves factual accuracy, and allows the system to answer questions about current or domain-specific information not present in the model's training data.

### Architecture

The RAG system follows a standard pipeline:
```
Load Documents → Chunk Text → Generate Embeddings → Store in Vector DB → Query → Retrieve → Generate Answer
```

### Components

#### Data Ingestion (`extract_text_from_url`)
- Fetches content from web URLs using the `requests` library
- Parses HTML with BeautifulSoup to extract readable text
- Removes non-content elements like scripts and styles
- Handles errors gracefully with timeout and exception handling

#### Text Chunking (`chunk_text`)
- Splits long documents into smaller, overlapping chunks (default: 300 words with 50-word overlap)
- Overlapping ensures context continuity across chunk boundaries
- Prevents loss of information at chunk edges

#### Embedding Generation (`load_embedding_model`)
- Uses SentenceTransformer's `all-MiniLM-L6-v2` model to convert text into vector embeddings
- Embeddings capture semantic meaning, enabling similarity-based retrieval
- Cached with Streamlit's `@st.cache_resource` for performance

#### Vector Database (`init_chroma`)
- Uses ChromaDB for efficient storage and retrieval of embeddings
- In-memory database for simplicity (persists during app session)
- Stores documents, embeddings, metadata, and unique IDs

#### Retrieval (`retrieve_chunks`)
- Converts user queries into embeddings
- Performs similarity search in the vector database
- Returns top-k most relevant text chunks (default: k=3)

#### Prompt Engineering (`build_prompt`)
- Constructs structured prompts for the LLM
- Includes retrieved context and clear instructions
- Enforces answer grounding: "Answer strictly based on context" or "I don't know"

#### Language Model Integration (`query_groq`)
- Uses Groq's Llama 3.1 8B Instant model for answer generation
- Low temperature (0.3) for more focused, factual responses
- Processes the augmented prompt to generate final answers

### User Interface

The Streamlit app provides an intuitive interface:

- **Sidebar**: URL input for document loading, document preview
- **Main Area**: Chat interface for querying the knowledge base
- **Session State**: Maintains loaded documents across interactions

### Tech Stack

- **Streamlit** - Web app framework
- **ChromaDB** - Vector database
- **SentenceTransformers** - Embedding model
- **Groq API** - LLM for answer generation
- **BeautifulSoup** - HTML parsing
- **Requests** - HTTP client

### Usage

1. Enter URLs in the sidebar text area (one per line)
2. Click "Load URLs" to process and store documents
3. Ask questions in the chat input
4. The system retrieves relevant context and generates grounded answers

This implementation demonstrates the fundamental principles of RAG systems and serves as a foundation for more advanced retrieval-augmented applications.

### Output
![alt text](/BasicRagSystem/image.png)