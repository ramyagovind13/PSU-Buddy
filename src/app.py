import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# app config
st.set_page_config(page_title="PSU Buddy", page_icon="🤖")
st.title("PSU Buddy")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]