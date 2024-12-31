# AgileKode Personal ChatBot

AgileKode Personal ChatBot is an intelligent Streamlit application designed to answer questions in two modes:
1. **Ask Me Anything:** A general-purpose chatbot for answering any query.
2. **Ask about AgileKode:** A chatbot powered by LangChain, vector databases, and Retrieval-Augmented Generation (RAG) architecture to provide accurate answers based on AgileKode's portfolio PDF.

The chatbot leverages LangChain for text processing, FAISS vector databases for efficient context-based retrieval, and RAG architecture for combining retrieval and generation, ensuring accurate and relevant responses.

---

## Features

- **Multi-Mode Chat:** Switch between general-purpose and AgileKode-specific modes.
- **PDF Parsing:** Extracts text from a PDF file for context-specific Q&A.
- **Vector Database:** Uses FAISS to store and retrieve text chunks from the PDF for efficient query processing.
- **LangChain Integration:** Manages text splitting and embeddings for enhanced query handling.
- **Retrieval-Augmented Generation (RAG):** Combines retrieved context with LLM capabilities to generate accurate, context-aware answers.
- **Intelligent Responses:** Leverages the Groq API to generate concise and meaningful responses.
- **Streamlit Interface:** Clean and interactive user interface for seamless interaction.

---

## RAG Architecture Overview

The chatbot employs a Retrieval-Augmented Generation (RAG) architecture to enhance its Q&A capabilities:

1. **Document Preprocessing:** The portfolio PDF is parsed, and its content is split into manageable chunks using LangChain.
2. **Embedding and Vectorization:** Text chunks are converted into embeddings using the SentenceTransformers model and stored in a FAISS vector database.
3. **Query Processing:** 
   - For AgileKode-specific queries, relevant text chunks are retrieved from the vector database based on similarity to the query.
   - These chunks provide context for generating answers.
4. **Answer Generation:** The Groq API generates answers by combining the retrieved context with the user query, ensuring responses are accurate and well-informed.

This approach ensures that answers are both grounded in the provided context and enriched with generative AI capabilities.
