import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


st.set_page_config(
    page_title="Banking Chatbot",
    page_icon="🏦",
    layout="centered"
)


@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    llm_model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)

    return embedding_model, tokenizer, llm_model


@st.cache_data
def load_dataset():
    with open("banking_data.json", "r", encoding="utf-8") as file:
        return json.load(file)


def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )


def find_best_answer(user_question, dataset, embedding_model, question_embeddings):
    user_embedding = embedding_model.encode(user_question)

    best_score = -1
    best_answer = ""

    for index, question_embedding in enumerate(question_embeddings):
        score = cosine_similarity(user_embedding, question_embedding)

        if score > best_score:
            best_score = score
            best_answer = dataset[index]["answer"]

    return best_answer, best_score


def generate_llm_answer(user_question, retrieved_answer, tokenizer, llm_model):
    prompt = f"""
You are a banking chatbot.

Use the information below to answer the question in one complete sentence.

Information:
{retrieved_answer}

Question:
{user_question}

Complete answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=80,
        min_new_tokens=15,
        num_beams=4,
        early_stopping=True
    )

    final_answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if len(final_answer.split()) < 5:
        final_answer = retrieved_answer

    return final_answer


st.title("🏦 Basic Banking Chatbot")
st.write("Ask basic banking questions using embeddings, retrieval, and a local LLM.")

dataset = load_dataset()
embedding_model, tokenizer, llm_model = load_models()

questions = [item["question"] for item in dataset]
question_embeddings = embedding_model.encode(questions)

user_question = st.text_input(
    "Enter your banking question:",
    placeholder="Example: What is KYC?"
)

if st.button("Ask"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        retrieved_answer, score = find_best_answer(
            user_question,
            dataset,
            embedding_model,
            question_embeddings
        )

        if score < 0.35:
            st.error("Sorry, I do not have enough banking information to answer that.")
        else:
            final_answer = generate_llm_answer(
                user_question,
                retrieved_answer,
                tokenizer,
                llm_model
            )

            st.success("Answer:")
            st.write(final_answer)

            with st.expander("View retrieval details"):
                st.write("Retrieved Banking Information:")
                st.info(retrieved_answer)
                st.write("Similarity Score:")
                st.write(round(float(score), 3))