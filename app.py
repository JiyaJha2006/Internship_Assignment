import uuid
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
st.set_page_config(
    page_title="Banking Chatbot",
    page_icon="🏦",
    layout="centered"
)
DB_PATH = "./vector_db"
BANKING_COLLECTION_NAME = "banking_knowledge_base"
MEMORY_COLLECTION_NAME = "chat_memory"
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    llm_model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
    return embedding_model, tokenizer, llm_model
@st.cache_resource
def load_vector_db():
    client = chromadb.PersistentClient(path=DB_PATH)
    banking_collection = client.get_or_create_collection(
        name=BANKING_COLLECTION_NAME
    )
    memory_collection = client.get_or_create_collection(
        name=MEMORY_COLLECTION_NAME
    )
    return banking_collection, memory_collection
embedding_model, tokenizer, llm_model = load_models()
banking_collection, memory_collection = load_vector_db()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_banking_context" not in st.session_state:
    st.session_state.latest_banking_context = ""
if "latest_memory_context" not in st.session_state:
    st.session_state.latest_memory_context = ""
if "latest_sources" not in st.session_state:
    st.session_state.latest_sources = []
if "latest_search_query" not in st.session_state:
    st.session_state.latest_search_query = ""
if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""
def is_follow_up_question(user_question):
    q = user_question.lower().strip()
    follow_up_phrases = [
        "it", "that", "this", "they", "them", "those", "these",
        "for it", "about it", "for that", "about that",
        "how much", "what documents", "documents needed",
        "required documents", "explain more", "tell me more",
        "what about", "does it", "can it", "is it"
    ]
    if len(q.split()) <= 5:
        return True
    for phrase in follow_up_phrases:
        if phrase in q:
            return True
    return False
def build_recent_chat_history():
    history_text = ""
    for chat in st.session_state.chat_history[-6:]:
        if chat["role"] == "user":
            history_text += f"User: {chat['message']}\n"
        else:
            history_text += f"Assistant: {chat['message']}\n"
    return history_text
def build_search_query(user_question):
    if st.session_state.current_topic and is_follow_up_question(user_question):
        return f"{st.session_state.current_topic}. {user_question}"
    return user_question
def extract_first_answer(banking_context):
    for line in banking_context.splitlines():
        line = line.strip()
        if line.startswith("Answer:"):
            return line.replace("Answer:", "").strip()
    return "I found relevant banking information, but could not extract a clear answer."
def extract_topic_from_question(user_question):
    return user_question.strip()
def retrieve_banking_context(search_query, embedding_model, banking_collection, top_k=3):
    query_embedding = embedding_model.encode(search_query).tolist()
    results = banking_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    context_text = ""
    for i in range(len(documents)):
        section = metadatas[i].get("section", "")
        question = metadatas[i].get("question", "")
        answer = metadatas[i].get("answer", "")
        context_text += f"""
Result {i + 1}:
Section: {section}
Question: {question}
Answer: {answer}
Distance: {distances[i]}
"""
    return context_text, metadatas, distances
def retrieve_chat_memory(search_query, embedding_model, memory_collection, top_k=1):
    if memory_collection.count() == 0:
        return ""
    query_embedding = embedding_model.encode(search_query).tolist()
    results = memory_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    documents = results["documents"][0]
    memory_text = ""
    for i, doc in enumerate(documents):
        memory_text += f"""
Past Conversation {i + 1}:
{doc}
"""
    return memory_text
def save_chat_memory(user_question, bot_answer, embedding_model, memory_collection):
    memory_text = f"""
User: {user_question}
Assistant: {bot_answer}
"""
    memory_embedding = embedding_model.encode(memory_text).tolist()
    memory_collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[memory_embedding],
        documents=[memory_text],
        metadatas=[
            {
                "user_question": user_question,
                "bot_answer": bot_answer
            }
        ]
    )
def choose_best_answer(user_question, banking_context):
    results = banking_context.split("Result ")

    user_words = set(
        user_question.lower()
        .replace("?", "")
        .replace(".", "")
        .replace(",", "")
        .split()
    )
    best_answer = ""
    best_score = -1
    important_words = [
        "document", "documents", "needed", "required", "proof",
        "aadhaar", "pan", "address", "interest", "rate",
        "savings", "account", "kyc", "loan", "atm", "deposit"
    ]
    for result in results:
        if "Answer:" not in result:
            continue
        lines = result.splitlines()
        question_text = ""
        answer_text = ""
        for line in lines:
            line = line.strip()
            if line.startswith("Question:"):
                question_text = line.replace("Question:", "").strip()
            if line.startswith("Answer:"):
                answer_text = line.replace("Answer:", "").strip()
        combined_text = (question_text + " " + answer_text).lower()
        score = 0
        for word in user_words:
            if word in combined_text:
                score += 2
        for word in important_words:
            if word in user_question.lower() and word in combined_text:
                score += 5
        if score > best_score:
            best_score = score
            best_answer = answer_text
    if best_answer == "":
        return extract_first_answer(banking_context)
    return best_answer
def generate_llm_answer(
    user_question,
    banking_context,
    memory_context,
    recent_chat_history,
    tokenizer,
    llm_model
):
    best_retrieved_answer = choose_best_answer(
        user_question,
        banking_context
    )
    prompt = f"""
You are a banking assistant.
Rewrite the retrieved answer into a short, clear answer for the user.
User Question:
{user_question}
Retrieved Answer:
{best_retrieved_answer}
Final Answer:
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
        num_beams=1,
        do_sample=False
    )
    final_answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()
    if (
        len(final_answer.split()) < 5
        or "?" in final_answer
        or final_answer.lower() == user_question.lower()
    ):
        final_answer = best_retrieved_answer
    return final_answer
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
st.markdown(
    '<div class="chat-title">🏦 Banking Chatbot</div>',
    unsafe_allow_html=True
)
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
    placeholder="Ask something like: What is a savings account?"
)
col1, col2 = st.columns([4, 1])
with col2:
    ask_button = st.button("Send", use_container_width=True)
if ask_button:
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        search_query = build_search_query(user_question)
        st.session_state.chat_history.append(
            {"role": "user", "message": user_question}
        )
        with st.spinner("Searching knowledge base and generating answer..."):
            banking_context, metadatas, distances = retrieve_banking_context(
                search_query,
                embedding_model,
                banking_collection,
                top_k=3
            )
            memory_context = retrieve_chat_memory(
                search_query,
                embedding_model,
                memory_collection,
                top_k=1
            )
            recent_chat_history = build_recent_chat_history()
            bot_reply = generate_llm_answer(
                user_question,
                banking_context,
                memory_context,
                recent_chat_history,
                tokenizer,
                llm_model
            )
            save_chat_memory(
                user_question,
                bot_reply,
                embedding_model,
                memory_collection
            )
            if not is_follow_up_question(user_question):
                st.session_state.current_topic = extract_topic_from_question(user_question)
            if st.session_state.current_topic == "":
                st.session_state.current_topic = extract_topic_from_question(user_question)
            st.session_state.latest_banking_context = banking_context
            st.session_state.latest_memory_context = memory_context
            st.session_state.latest_sources = metadatas
            st.session_state.latest_search_query = search_query
        st.session_state.chat_history.append(
            {"role": "bot", "message": bot_reply}
        )

        st.rerun()
with st.expander("View Contextual Search Query"):
    if st.session_state.latest_search_query == "":
        st.write("Ask a question first.")
    else:
        st.write(st.session_state.latest_search_query)
with st.expander("View Latest Retrieved Banking Context"):
    if st.session_state.latest_banking_context == "":
        st.write("Ask a question first to view retrieval details.")
    else:
        st.write(st.session_state.latest_banking_context)
with st.expander("View Retrieved Chat Memory"):
    if st.session_state.latest_memory_context == "":
        st.write("No relevant past memory retrieved yet.")
    else:
        st.write(st.session_state.latest_memory_context)
with st.expander("View Retrieved Source Questions"):
    if len(st.session_state.latest_sources) == 0:
        st.write("No sources retrieved yet.")
    else:
        for i, source in enumerate(st.session_state.latest_sources):
            st.write(f"Source {i + 1}")
            st.write("Section:", source.get("section", ""))
            st.write("Question:", source.get("question", ""))
            st.write("Answer:", source.get("answer", ""))
with st.expander("Current Topic"):
    if st.session_state.current_topic == "":
        st.write("No current topic yet.")
    else:
        st.write(st.session_state.current_topic)
col_clear1, col_clear2 = st.columns(2)
with col_clear1:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.latest_banking_context = ""
        st.session_state.latest_memory_context = ""
        st.session_state.latest_sources = []
        st.session_state.latest_search_query = ""
        st.session_state.current_topic = ""
        st.rerun()
with col_clear2:
    if st.button("Clear Persistent Memory"):
        all_memory = memory_collection.get()
        if len(all_memory["ids"]) > 0:
            memory_collection.delete(ids=all_memory["ids"])
        st.session_state.chat_history = []
        st.session_state.latest_banking_context = ""
        st.session_state.latest_memory_context = ""
        st.session_state.latest_sources = []
        st.session_state.latest_search_query = ""
        st.session_state.current_topic = ""
        st.rerun()
