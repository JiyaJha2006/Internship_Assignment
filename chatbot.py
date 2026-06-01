import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


with open("banking_data.json", "r", encoding="utf-8") as file:
    dataset = json.load(file)


print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded successfully!")


print("Loading local LLM...")
llm_model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)

print("Local LLM loaded successfully!")


questions = [item["question"] for item in dataset]
question_embeddings = embedding_model.encode(questions)


def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )


def find_best_answer(user_question):
    user_embedding = embedding_model.encode(user_question)

    best_score = -1
    best_answer = ""

    for index, question_embedding in enumerate(question_embeddings):
        score = cosine_similarity(user_embedding, question_embedding)

        if score > best_score:
            best_score = score
            best_answer = dataset[index]["answer"]

    return best_answer, best_score


def generate_llm_answer(user_question, retrieved_answer):
    prompt = f"""
You are a banking chatbot.

Use the information below to answer the question in one complete sentence.

Retrieved Banking Information:
{retrieved_answer}

Question:
{user_question}

Give a complete answer:
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


print("\nBanking Chatbot is ready!")
print("Ask a banking question.")
print("Type 'exit' to stop.\n")


while True:
    user_question = input("You: ")

    if user_question.lower() == "exit":
        print("Bot: Goodbye!")
        break

    retrieved_answer, score = find_best_answer(user_question)

    if score < 0.35:
        print("Bot: Sorry, I do not have enough banking information to answer that.")
    else:
        final_answer = generate_llm_answer(user_question, retrieved_answer)
        print("Bot:", final_answer)

    print()