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
dataset = load_dataset()
embedding_model, tokenizer, llm_model = load_models()
questions = [item["question"] for item in dataset]
question_embeddings = embedding_model.encode(questions)
st.markdown("""
<style>
.stApp {
    background-color: #343541;
}
.block-container {
    max-width: 850px;
    padding-top: 2rem;
}
.chat-title {
    text-align: center;
    color: #ffffff;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}
.chat-subtitle {
    text-align: center;
    color: #c5c5d2;
    font-size: 16px;
    margin-bottom: 35px;
}
.user-message {
    background-color: #2f80ed;
    color: white;
    padding: 16px 20px;
    border-radius: 18px 18px 4px 18px;
    margin: 15px 0 15px auto;
    max-width: 75%;
    font-size: 16px;
}
.bot-message {
    background-color: #444654;
    color: #ffffff;
    padding: 16px 20px;
    border-radius: 18px 18px 18px 4px;
    margin: 15px auto 15px 0;
    max-width: 75%;
    font-size: 16px;
}
.stTextInput input {
    background-color: #40414f !important;
    color: white !important;
    border: 1px solid #565869 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
.stTextInput label {
    color: #ffffff !important;
}
.stButton button {
    background-color: #10a37f !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    height: 45px;
    font-weight: 600;
}
.stButton button:hover {
    background-color: #0e8f70 !important;
}
[data-testid="stExpander"] {
    background-color: #40414f !important;
    border: 1px solid #565869 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] * {
    color: white !important;
}
.footer {
    text-align: center;
    color: #c5c5d2;
    margin-top: 30px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="chat-title">🏦 Banking Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="chat-subtitle">Ask banking questions using Embeddings, Retrieval, and a Local LLM</div>',
    unsafe_allow_html=True
)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(
            f'<div class="user-message">{chat["message"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="bot-message">{chat["message"]}</div>',
            unsafe_allow_html=True
        )
user_question = st.text_input(
    "Message Banking Chatbot",
    placeholder="Ask something like: What is KYC?"
)
col1, col2 = st.columns([4, 1])
with col2:
    ask_button = st.button("Send", use_container_width=True)
if ask_button:
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        st.session_state.chat_history.append(
            {"role": "user", "message": user_question}
        )
        with st.spinner("Thinking..."):
            retrieved_answer, score = find_best_answer(
                user_question,
                dataset,
                embedding_model,
                question_embeddings
            )
            if score < 0.35:
                bot_reply = "Sorry, I do not have enough banking information to answer that."
            else:
                bot_reply = generate_llm_answer(
                    user_question,
                    retrieved_answer,
                    tokenizer,
                    llm_model
                )
        st.session_state.chat_history.append(
            {"role": "bot", "message": bot_reply}
        )
        st.rerun()
with st.expander("🔍 View Latest Retrieval Details"):
    if len(st.session_state.chat_history) == 0:
        st.write("Ask a question first to view retrieval details.")
    else:
        last_user_question = None

        for chat in reversed(st.session_state.chat_history):
            if chat["role"] == "user":
                last_user_question = chat["message"]
                break
        if last_user_question:
            retrieved_answer, score = find_best_answer(
                last_user_question,
                dataset,
                embedding_model,
                question_embeddings
            )
            st.write("Retrieved Banking Information:")
            st.info(retrieved_answer)
            st.write("Similarity Score:")
            st.write(round(float(score), 3))
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
st.markdown(
    '<div class="footer">Built using Sentence Transformers + FLAN-T5 + Streamlit</div>',
    unsafe_allow_html=True
)
