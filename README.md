# Basic Banking Chatbot using LLM and Embeddings

## Overview

This project is a simple Retrieval-Augmented Generation (RAG) based Banking Chatbot developed as part of an AI Internship Assignment.

The chatbot answers basic banking-related questions by:

1. Converting user questions into embeddings.
2. Retrieving the most relevant banking information from a predefined dataset.
3. Passing the retrieved information and user query to a Local Language Model (LLM).
4. Generating a natural language response.

---

## Features

* Banking Question Answering
* Embedding-based Information Retrieval
* Cosine Similarity Search
* Local LLM Integration
* Command Line Interface
* Lightweight and Easy to Run

---

## Technologies Used

### Embedding Model

* Sentence Transformers
* Model: `all-MiniLM-L6-v2`

### Language Model (LLM)

* Google FLAN-T5 Base
* Model: `google/flan-t5-base`

### Programming Language

* Python 3

### Libraries

* sentence-transformers
* transformers
* torch
* numpy
* sentencepiece

---

## Project Structure


banking_chatbot/
│
├── chatbot.py
├── banking_data.json
├── requirements.txt
└── README.md


---

## Working Architecture


User Question
      │
      ▼
Embedding Generation
(all-MiniLM-L6-v2)
      │
      ▼
Cosine Similarity Search
      │
      ▼
Retrieve Most Relevant Banking Information
      │
      ▼
Pass Query + Retrieved Context to FLAN-T5
      │
      ▼
Generate Final Response
      │
      ▼
Display Answer to User


---

## Dataset

The chatbot uses a small banking knowledge base stored in JSON format.

Sample Topics:

* KYC (Know Your Customer)
* Savings Account
* Current Account
* Fixed Deposit
* Internet Banking
* ATM Card
* Loan
* Bank Account Opening Documents

---

## Retrieval Process

1. All banking questions are converted into embeddings.
2. The user's query is converted into an embedding.
3. Cosine similarity is calculated between the user query and stored questions.
4. The most relevant banking answer is retrieved.

---

## LLM Response Generation

After retrieval:

* The retrieved banking information is combined with the user query.
* The prompt is passed to FLAN-T5.
* FLAN-T5 generates the final response in natural language.

This follows the Retrieval-Augmented Generation (RAG) approach.

---

## Installation

### Clone Repository


git clone <repository-link>
cd banking_chatbot


### Install Dependencies


pip install -r requirements.txt


---

## Running the Chatbot


python chatbot.py


---

## Example


You: What is KYC?

Bot: KYC means Know Your Customer. It is a process used by banks to verify the identity and address of customers.



You: What documents are required to open a bank account?

Bot: Common documents required are Aadhaar card, PAN card, address proof, identity proof, and passport-size photographs.


---

## Future Improvements

* Larger Banking Knowledge Base
* Web Interface using Flask or Streamlit
* Vector Database Integration
* Multi-turn Conversations
* Fine-tuned Banking LLM
* Voice-based Interaction

---

## Conclusion

This project demonstrates the use of Embeddings, Retrieval, and Large Language Models to build a simple Banking Question Answering system. It implements a basic Retrieval-Augmented Generation (RAG) pipeline using Sentence Transformers and FLAN-T5.

---

This project demonstrates the use of Embeddings, Retrieval, and Large Language Models to build a simple Banking Question Answering system. It implements a basic Retrieval-Augmented Generation (RAG) pipeline using Sentence Transformers and FLAN-T5.
