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

vector_store = QdrantVectorStore(client=qdrant_client, collection_name="zh-data-example")
# Create a VectorStoreIndex from the documents
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
# Query the index
query_engine = index.as_query_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("what's up?"):
    with st.chat_message(name="user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role":"user", "content":prompt})
    final_prompt = f"""
    你現在是一位人類圖專家幫助管理者(Dia)為顧客提供有關人類圖相關的問題解答，請根據顧客的問題：{prompt}，
    你可以忽略根據你的現有知識提供有關人類圖的內容，如果顧客詢問不相關的問題請一律回覆：我無法回答此問題，請嘗試新的問題。
    """

    response = query_engine.query(final_prompt)

    with st.chat_message(name="assistent"):
        st.markdown(response)
    st.session_state.messages.append({"role":"assistent", "content":response})