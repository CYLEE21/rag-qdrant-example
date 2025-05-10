from dotenv import load_dotenv 
import os

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import streamlit as st

load_dotenv(".env")
GEMINI_KEY = os.environ.get("GEMINI_KEY")
QDRANT_KEY = os.environ.get("QDRANT_KEY")
COLLECTION = os.environ.get("COLLECTION")

gemini_model = "models/gemini-2.0-flash"
embed_model = "models/gemini-embedding-exp-03-07"

# Connect service client.
qdrant_client = QdrantClient(
    url="https://2fca434a-57ac-427f-9488-377cd4093eaa.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=QDRANT_KEY
)

# Setting the llama_index contect models.
Settings.llm = Gemini(model=gemini_model,api_key=GEMINI_KEY, temperature=0.5)
Settings.embed_model = GeminiEmbedding(
    model_name=embed_model, api_key=GEMINI_KEY, title="this is a document"
)

vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION)
# Create a VectorStoreIndex from the documents
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
# Query the index
query_engine = index.as_query_engine(similarity_top_k=5)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me something..."):
    with st.chat_message(name="user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role":"user", "content":prompt})

    response = query_engine.query(prompt)

    with st.chat_message(name="assistent"):
        st.markdown(response)
    st.session_state.messages.append({"role":"assistent", "content":response})