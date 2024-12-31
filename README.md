AgileKode Personal ChatBot
Welcome to the AgileKode Personal ChatBot, a Streamlit-based AI chatbot capable of answering general questions or providing specific insights about the AgileKode portfolio. This project demonstrates an integration of advanced LLM models with retrieval-augmented generation (RAG) techniques for enhanced contextual responses.

Features
Two Interaction Modes:
Ask Me Anything: General-purpose Q&A powered by Groq's LLM.
Ask about AgileKode: Provides insights into the AgileKode portfolio using a preprocessed PDF file for context.
PDF Knowledge Extraction:
Processes the agilekode-portfolio.pdf file to create a vector database using FAISS and SentenceTransformers.
Retrieval-Augmented Generation (RAG):
Dynamically retrieves relevant excerpts from the vector database to generate accurate, context-aware responses.
Session History:
Maintains chat history for both modes, displayed interactively on the Streamlit interface.
Interactive UI:
Built with Streamlit, offering a clean and responsive user experience.

Installation
Prerequisites
Python 3.8 or later
Required Python packages listed in requirements.txt
