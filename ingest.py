import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
CSV_FILE = "banking_knowledge_base_1000.csv"
DB_PATH = "./vector_db"
COLLECTION_NAME = "banking_knowledge_base"
print("Loading dataset...")
df = pd.read_csv(CSV_FILE)
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)
print("Preparing documents...")
ids = []
documents = []
metadatas = []
embeddings = []
for index, row in df.iterrows():
    section = str(row["Section"])
    question = str(row["Question"])
    answer = str(row["Answer"])
    document_text = f"""
Section: {section}
Question: {question}
Answer: {answer}
"""
    embedding = embedding_model.encode(document_text).tolist()
    ids.append(str(index))
    documents.append(document_text)
    embeddings.append(embedding)
    metadatas.append(
        {
            "section": section,
            "question": question,
            "answer": answer
        }
    )
print("Adding documents to ChromaDB...")
collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)
print("Vector database created successfully!")
print("Total documents stored:", collection.count())