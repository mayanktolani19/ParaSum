import streamlit as st
from summarizer import text_summarize

def app():
    st.title('Summarizer')
    st.write('Please provide the text to be summarized')
    user_input = st.text_area('Enter text','')
    minLength = st.slider('Min Words required in summary',20,100,35)
    maxLength = st.slider('Max Words required in summary',80,350,120)
    output = text_summarize(user_input,minLength,maxLength)
    st.write("Text Summary: ")
    st.write(output)