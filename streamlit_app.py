import os
import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF

os.environ["USE_TF"] = "0"  # Prevent TensorFlow issues


st.title("🧠 Simple QA Bot")

# Load the QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

question = st.text_input("Ask your question:")
context = st.text_area("Context:", "This is a simple example of a context where the answer might be found.")

if st.button("Get Answer") and question and context:
    result = qa_pipeline(question=question, context=context)
    st.write("**Answer:**", result["answer"])
