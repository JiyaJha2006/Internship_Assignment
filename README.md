# 🏦 Banking Chatbot using Embeddings, Retrieval-Augmented Generation (RAG), and Local LLM

## Overview

This project is a Banking Chatbot developed using Retrieval-Augmented Generation (RAG).

The chatbot combines:

* Sentence Transformer Embeddings
* Cosine Similarity Retrieval
* FLAN-T5 Local Language Model
* Streamlit Web Interface

The system retrieves relevant banking information from a knowledge base and uses a local LLM to generate natural language responses.

---

# Features

### AI Features

* Semantic Search using Embeddings
* Retrieval-Augmented Generation (RAG)
* Local Large Language Model (FLAN-T5)
* Banking Question Answering
* Similarity-based Information Retrieval

### User Interface Features

* ChatGPT-style Interface
* Conversation History
* User and Bot Chat Bubbles
* Real-time Response Generation
* Retrieval Details Viewer
* Clear Chat Button
* Responsive Web Interface

---

# Technologies Used

## Frontend

* Streamlit
* HTML/CSS Styling
* ChatGPT-inspired UI Design

## Backend

* Python
* Sentence Transformers
* Transformers Library
* NumPy

## Language Models

### Embedding Model

```text
all-MiniLM-L6-v2
```

Used for converting user questions and banking questions into vector embeddings.

### Local LLM

```text
google/flan-t5-base
```

Used for generating natural language responses.

---

# Project Structure

```text
banking_chatbot/
│
├── app.py
├── chatbot.py
├── banking_data.json
├── requirements.txt
└── README.md
```

---

# System Architecture

```text
User Question
        │
        ▼
Frontend (Streamlit UI)
        │
        ▼
Sentence Transformer
(all-MiniLM-L6-v2)
        │
        ▼
Embedding Generation
        │
        ▼
Cosine Similarity Search
        │
        ▼
Retrieve Banking Context
        │
        ▼
FLAN-T5 Local LLM
        │
        ▼
Generate Final Answer
        │
        ▼
Display Response in UI
```

---

# Backend Workflow

### Step 1: User Query

The user enters a banking question through the Streamlit interface.

Example:

```text
What is KYC?
```

---

### Step 2: Embedding Generation

The query is converted into a vector representation using:

```text
all-MiniLM-L6-v2
```

---

### Step 3: Similarity Search

Cosine similarity is calculated between:

* User Query Embedding
* Stored Banking Question Embeddings

The most relevant banking question is identified.

---

### Step 4: Retrieval

The corresponding banking answer is retrieved from:

```text
banking_data.json
```

---

### Step 5: LLM Generation

The system sends:

* User Question
* Retrieved Banking Information

to:

```text
FLAN-T5 Base
```

The LLM generates the final response.

---

### Step 6: Display Response

The generated answer is displayed in the chatbot interface.

---

# Frontend Workflow

The frontend is built using Streamlit.

Components used:

### Chat Window

Displays:

* User Messages
* Bot Responses

using chat bubbles.

### Input Box

Allows users to enter banking questions.

### Send Button

Triggers backend processing.

### Retrieval Details Section

Shows:

* Retrieved Banking Information
* Similarity Score

for transparency and debugging.

### Clear Chat Button

Clears the conversation history.

---

# Banking Knowledge Base

The chatbot currently supports:

* KYC
* Savings Account
* Current Account
* Fixed Deposit
* ATM Card
* Internet Banking
* Loans
* Account Opening Documents

The knowledge base is stored in:

```text
banking_data.json
```

---

# Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the Application

Launch the Streamlit interface:

```bash
python -m streamlit run app.py
```

---

# Example Conversation

```text
User:
What is KYC?

Bot:
KYC means Know Your Customer. It is a process used by banks to verify the identity and address of customers.
```

---

# Future Improvements

* Larger Banking Dataset
* PDF Knowledge Base Support
* Vector Database Integration (FAISS/Pinecone)
* Voice-Based Chatbot
* Multi-turn Context Memory
* Fine-tuned Banking LLM
* Authentication System

---

# Conclusion

This project demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline using Sentence Transformers and FLAN-T5. The application combines semantic retrieval, local language model generation, and a modern Streamlit-based ChatGPT-style user interface to provide banking-related question answering.
