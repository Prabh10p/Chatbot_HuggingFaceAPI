from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id ="google/gemma-2-9b-it",
    task = "conversation"
)

model = ChatHuggingFace(llm = llm)

st.header("Chat Bot Assistant")
st.markdown("<style>div.stApp {background: url('https://picsum.photos/1920/1080') no-repeat center center fixed; background-size: cover;}</style>", unsafe_allow_html=True)

user_input = st.text_input("What do you want to ask? ")

topic_input = st.selectbox("Choose a topic",["Technology","Health","Finance","Education","Entertainment"])

style_input = st.selectbox("Choose a style",["Formal","Informal","Humorous","Serious","Casual"])

length_input = st.selectbox("Choose length",["Short","Medium","Long"])



prompt = PromptTemplate(
    inmput_variables = ["user_input","topic_input","style_input","lenght_input"],
    template = "You are expert in {topic_input}. Answer {user_input} in a {style_input} with {length_input} response")



user_input = prompt.invoke(
    {
        "user_input":user_input,
        "topic_input":topic_input,
        "style_input":style_input,
        "length_input":length_input
    }
)

if st.button("Submit"):
    response = model.invoke(user_input)
    st.write(response.content)


# Email - p0l81532@marymout.edu