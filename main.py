import streamlit as st
from multiapp import MultiApp
# import your app modules here
from apps import paraphraseApp, summarizeApp

app = MultiApp()

st.markdown("""
# ParaSum

ParaSum provides two services - Text Paraphrasing and Text Summarizing. It utilizes HuggingFace transformer models for both the tasks. 

## Enter your text and see the magic!

""")

# Add all your application here
app.add_app("Paraphraser", paraphraseApp.app)
app.add_app("Summarizer", summarizeApp.app)
# The main app
app.run()
