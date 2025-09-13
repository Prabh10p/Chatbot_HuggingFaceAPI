import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Get token: local env fallback or Streamlit Cloud secret
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="conversational" # <- must be passed here
)

model = ChatHuggingFace(llm=llm)


if 'chat_history' not in st.session_state:
    st.session_state.chat_history=[]
 
# Streamlit UI
st.header("Chat Bot Assistant")
user_input = st.text_input("You: ")
style_input = st.selectbox("Choose a style", ["Formal", "Informal", "Humorous", "Serious", "Casual"])
length_input = st.selectbox("Choose length", ["Short", "Medium", "Long"])

if st.button("Answer") and user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    prompt = f"You are helpful assistant. Answer '{user_input}' in a {style_input} style and keep answer {length_input}."
    response = model.invoke(st.session_state.chat_history + [HumanMessage(content=prompt)])
    st.session_state.chat_history.append(response)
    st.write(response.content)
