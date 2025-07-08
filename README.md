# Streamlit LLM Applications

This repository contains a collection of Streamlit applications demonstrating various functionalities of Large Language Models (LLMs) using LangChain. These applications cover Retrieval Augmented Generation (RAG) for document querying, conversational chatbots, and URL content summarization.

## Table of Contents

1. Features

2. Applications

   - 1. RAG Document Q&A with Groq (`app.py`)
     
   - 2. Conversational RAG with PDF Uploads and Chat History (`app.py`)

   - 3. Ollama Q&A Chatbot (`app.py`)

   - 4. Q&A Chatbot with OpenAI (`app.py`)

   - 5. Langchain: URL Content Summarizer (`app.py`)

3. Setup and Installation

   - Prerequisites

   - Environment Variables and API Keys

   - Installation

   - Running the Applications

4. Contributing

5. License
   
## Features

- **Retrieval Augmented Generation (RAG):** Query custom documents (PDFs) to get context-aware answers.

- **Conversational Chatbots:** Engage in multi-turn conversations with LLMs, maintaining chat history.

- **Multiple LLM Integrations:** Support for Groq, OpenAI, and Ollama models.

- **Various Embedding Models:** Utilize OpenAI Embeddings and HuggingFace Embeddings.

- **Vector Store Options:** Integration with FAISS and ChromaDB for efficient document retrieval.

- **URL Content Summarization:** Summarize content from web pages and YouTube videos.

- **Streamlit UI:** Interactive and user-friendly web interfaces for all applications.

## Applications

### 1. RAG Document Q&A with Groq (`app.py`)

This application allows you to perform Question Answering on a collection of PDF research papers using a RAG system. It leverages Groq's LLM for generating answers and OpenAI Embeddings for creating document vectors.

- **Functionality:**

  - Loads PDF documents from a `research_papers` directory.

  - Splits documents into chunks and creates vector embeddings.

  - Stores embeddings in a FAISS vector database.

  - Answers user queries based on the retrieved context from the PDFs.

  - Displays the relevant document chunks used for generating the answer.

- **Key Technologies:**

  - **LLM:** `ChatGroq` (using `gemma2-9b-it`)

  - **Embeddings:** `OpenAIEmbeddings`

  - **Vector Store:** `FAISS`

  - **Document Loader:** `PyPDFDirectoryLoader`

  - **Text Splitter:** `RecursiveCharacterTextSplitter`

  - **LangChain Chains:** `create_stuff_documents_chain`, `create_retrieval_chain`

- **Setup Notes:**

  - Create a directory named `research_papers` in the same location as `app_1.py` and place your PDF files inside it.

  - Requires `OPENAI_API_KEY` and `GROQ_API_KEY`.

### 2. Conversational RAG with PDF Uploads and Chat History (`app.py`)

This is a more advanced RAG application that supports uploading PDF files directly through the Streamlit interface and maintains a conversational chat history.

- **Functionality:**

  - Allows users to upload multiple PDF documents.

  - Processes uploaded PDFs to create a knowledge base.

  - Engages in multi-turn conversations, remembering previous interactions.

  - Contextualizes new questions based on chat history before retrieval.

- **Key Technologies:**

  - **LLM:** `ChatGroq` (using `Gemma2-9b-It`)

  - **Embeddings:** `HuggingFaceEmbeddings` (using `all-MiniLM-L6-v2`)

  - **Vector Store:** `Chroma`

  - **Document Loader:** `PyPDFLoader`

  - **Text Splitter:** `RecursiveCharacterTextSplitter`

  - **LangChain Components:** `create_history_aware_retriever`, `create_retrieval_chain`, `create_stuff_documents_chain`, `RunnableWithMessageHistory`, `ChatMessageHistory`, `MessagesPlaceholder`

- **Setup Notes:**

  - Requires `GROQ_API_KEY` and `HF_TOKEN`.

### 3. Ollama Q&A Chatbot (`app.py`)

A simple Q&A chatbot designed to interact with locally run Ollama models.

- **Functionality:**

  - Provides a chat interface for asking questions.

  - Allows selection of different Ollama models (e.g., `gemma2:2b`, `llama2`, `mistral`).

  - Adjustable parameters: temperature and max response length.

  - Maintains basic chat history in the UI.

- **Key Technologies:**

  - **LLM:** `Ollama`

  - **LangChain Components:** `ChatPromptTemplate`, `StrOutputParser`

- **Setup Notes:**

  - Requires [Ollama](https://ollama.com/) to be installed and the desired models pulled (e.g., `ollama pull gemma2:2b`).

  - Optionally uses `LANGCHAIN_API_KEY` for Langsmith tracing.

### 4. Q&A Chatbot with OpenAI (`app.py`)

A general-purpose chatbot that leverages OpenAI's powerful language models.

- **Functionality:**

  - Provides a chat interface.

  - Allows selection of various OpenAI models (e.g., `gpt-4o`, `gpt-4-turbo`, `gpt-4`).

  - Adjustable parameters: temperature and max tokens.

  - Maintains conversational history.

- **Key Technologies:**

  - **LLM:** `ChatOpenAI`

  - **LangChain Components:** `ChatPromptTemplate`, `MessagesPlaceholder`, `StrOutputParser`

- **Setup Notes:**

  - Requires `OPENAI_API_KEY`.

### 5. Langchain: URL Content Summarizer (`app.py`)

This application summarizes content from any given URL, including YouTube video transcripts.

- **Functionality:**

  - Accepts a URL (web page or YouTube video).

  - Loads content using appropriate loaders (`UnstructuredURLLoader` for web, `YoutubeLoader` for YouTube).

  - Generates a concise summary using a Groq LLM.

- **Key Technologies:**

  - **LLM:** `ChatGroq` (using `gemma2-9b-it`)

  - **Loaders:** `YoutubeLoader`, `UnstructuredURLLoader`

  - **LangChain Chain:** `load_summarize_chain` (with `stuff` chain type)

  - **Validation:** `validators` library for URL validation.

- **Setup Notes:**

  - Requires `GROQ_API_KEY`.

## Setup and Installation

### Prerequisites

- Python 3.8+

- `pip` (Python package installer)

### Environment Variables and API Keys

Many of these applications require API keys for various services (Groq, OpenAI, Hugging Face). It is highly recommended to set these up as environment variables.

1. Create a file named `.env` in the root directory of your project (where your `app.py` files are located).

2. Add your API keys and other environment variables to this file. Replace the placeholder values with your actual keys:
```

OPENAI_API_KEY="your_openai_api_key_here"
GROQ_API_KEY="your_groq_api_key_here"
HF_TOKEN="your_huggingface_token_here" \# For HuggingFaceEmbeddings
LANGCHAIN_API_KEY="your_langchain_api_key_here" \# Optional, for Langsmith tracing
LANGCHAIN_TRACING_V2="true" \# Optional, for Langsmith tracing
LANGCHAIN_PROJECT="Your_Langsmith_Project_Name" \# Optional, for Langsmith tracing

```

* **Groq API Key:** Get it from [Groq Console](https://console.groq.com/keys).

* **OpenAI API Key:** Get it from [OpenAI API Keys](https://platform.openai.com/api-keys).

* **Hugging Face Token:** Get it from [Hugging Face Settings](https://huggingface.co/settings/tokens).

### Installation

1. **Clone the repository (if applicable) or download the files.**

2. **Navigate to the project directory:**

```

cd /path/to/your/project

```

3. **Create a virtual environment (recommended):**

```

python -m venv venv
source venv/bin/activate \# On Windows: venv\\Scripts\\activate

```

4. **Install the required packages:**
Create a `requirements.txt` file with the following content:

```

streamlit\>=1.30.0
langchain\>=0.2.0
langchain-core\>=0.2.0
langchain-community\>=0.2.0
langchain-openai\>=0.1.0
langchain-groq\>=0.1.0
langchain-huggingface\>=0.0.3 
langchain-chroma\>=0.2.0 
langchain-text-splitters\>=0.2.0
python-dotenv\>=1.0.0
pypdf\>=4.0.0 \# For PDF loading
faiss-cpu\>=1.7.0 \
ollama \# (Note: Ollama itself needs to be installed separately)
validators\>=0.20.0 \
unstructured[pdf] \# (if using UnstructuredURLLoader with PDFs)
youtube-transcript-api \# (YoutubeLoader)

```

Then install:

```

pip install -r requirements.txt

```

### Running the Applications

To run any of the Streamlit applications, use the `streamlit run` command followed by the script name:

For `app.py`:

```

streamlit run app.py

```


The application will open in your web browser.

## Contributing

Contributions are welcome! If you have ideas for new applications, improvements, or bug fixes, feel free to open an issue or submit a pull request.

## License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).
```
