# 🏦 Banking Chatbot using RAG, ChromaDB, and Local LLM

## 📌 Project Overview

This project is an intelligent Banking Chatbot built using Retrieval-Augmented Generation (RAG).

The chatbot answers banking-related questions by retrieving relevant information from a banking knowledge base stored in a ChromaDB vector database and generating natural language responses using a local Large Language Model (FLAN-T5).

Unlike traditional chatbots, this system supports:

* Semantic search using embeddings
* Persistent vector storage using ChromaDB
* Follow-up question handling
* Context-aware retrieval
* Conversation memory
* Local LLM inference
* Interactive Streamlit web interface

---

# 🚀 Features

## 1. Banking Knowledge Base

The chatbot uses a large banking dataset containing questions and answers related to:

* Savings Accounts
* Current Accounts
* Fixed Deposits
* Loans
* KYC
* ATM Services
* Debit Cards
* Credit Cards
* Online Banking
* Mobile Banking
* Account Opening Procedures

---

## 2. Semantic Search using Sentence Transformers

Model Used:

all-MiniLM-L6-v2

Instead of keyword matching, user queries are converted into vector embeddings.

Example:

User Question:

What documents are required for opening an account?

Even if the dataset contains:

What documents are needed to open a bank account?

the chatbot can still retrieve the correct answer because the semantic meaning is similar.

---

## 3. ChromaDB Vector Database

All banking knowledge is stored inside a persistent ChromaDB vector database.

Benefits:

* Fast retrieval
* Semantic search
* Persistent storage
* Scalable architecture

Workflow:

Dataset
↓
Embeddings
↓
ChromaDB
↓
Retriever
↓
Answer Generation

---

## 4. Persistent Conversation Memory

The chatbot stores previous conversations inside a separate ChromaDB memory collection.

Stored Format:

User: What is KYC?
Assistant: KYC is the process used by banks to verify customer identity.

Benefits:

* Remembers previous discussions
* Retrieves relevant past conversations
* Supports follow-up questions

---

## 5. Context-Aware Retrieval

The chatbot tracks the current topic.

Example:

User: What is a savings account?

User: How much interest does it give?

User: What documents are needed for it?

The chatbot understands that:

"it" = savings account

and retrieves information related to savings accounts instead of searching the entire dataset blindly.

---

## 6. Local Large Language Model

Model Used:

google/flan-t5-base

Purpose:

* Rewrite retrieved information
* Improve readability
* Generate conversational responses

Important:

The model does NOT generate answers from its own knowledge.

Instead:

Retrieve Information
↓
Select Best Match
↓
Rewrite Response
↓
Display Answer

This reduces hallucinations and improves accuracy.

---

## 7. Best Answer Selection

When multiple documents are retrieved:

Result 1
Result 2
Result 3

the chatbot automatically scores them against the user question and selects the most relevant answer.

This prevents unrelated answers from being returned.

---

## 8. Streamlit Web Interface

The application includes a modern conversational UI inspired by ChatGPT.

Features:

* Chat interface
* Dark theme
* Conversation history
* Source inspection
* Memory inspection
* Context query visualization

---

# 🏗 System Architecture

User Question
↓
Context Builder
↓
Sentence Transformer
↓
Embedding Vector
↓
ChromaDB Retrieval
↓
Top-K Results
↓
Best Answer Selector
↓
FLAN-T5 Rewriter
↓
Final Response

---

# 🧠 Conversation Memory Architecture

User Question
↓
Embedding
↓
Memory Vector Database
↓
Retrieve Similar Conversations
↓
Add Context
↓
Generate Answer

---

# 📂 Project Structure

banking_chatbot/

├── app.py

├── banking_data.csv

├── create_vector_db.py

├── vector_db/

├── requirements.txt

└── README.md

---

# ⚙️ Installation

Install dependencies:

pip install -r requirements.txt

---

# ▶️ Running the Application

Start Streamlit:

streamlit run app.py

Application will open at:

http://localhost:8501

---

# 📦 Technologies Used

Frontend:

* Streamlit

Backend:

* Python

Embeddings:

* Sentence Transformers
* all-MiniLM-L6-v2

Vector Database:

* ChromaDB

Language Model:

* FLAN-T5 Base

Machine Learning Libraries:

* Transformers
* PyTorch

---

# Future Improvements

* Source citations below answers
* Stronger local LLM (Phi-3 Mini / Qwen 2.5)
* LangChain integration
* Multi-turn reasoning
* PDF knowledge ingestion
* Banking document upload support
* Voice-enabled chatbot

---

# Learning Outcomes

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Embedding Models
* Semantic Search
* Local LLM Deployment
* Conversational Memory
* Context-Aware Retrieval
* Streamlit Application Development

---
