import streamlit as st
from htmlTemplates import css, bot_template, user_template

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def pdf_chat_app(session_vars_1 = st.session_state.conversation):
    
    if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

    st.write(css, unsafe_allow_html=True)
    st.header("Start asking questions on the selected document :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)