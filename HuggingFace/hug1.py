import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Get token: local env fallback or Streamlit Cloud secret
import os
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN") or st.secrets.get("HUGGINGFACE_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="conversational",
    api_key=HF_TOKEN   # <- must be passed here
)

model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.header("Chat Bot Assistant")
user_input = st.text_input("What do you want to ask?")
topic_input = st.selectbox("Choose a topic", ["Technology", "Health", "Finance", "Education", "Entertainment"])
style_input = st.selectbox("Choose a style", ["Formal", "Informal", "Humorous", "Serious", "Casual"])
length_input = st.selectbox("Choose length", ["Short", "Medium", "Long"])

if st.button("Answer"):
    # Make sure you pass a string, not a Streamlit object
    prompt = f"You are an expert in {topic_input}. Answer '{user_input}' in a {style_input} style and keep answer {length_input}."
    response = model.invoke(prompt)
    st.write(response.content)
