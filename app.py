import streamlit as st
import pickle

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Sentiment Analysis using Naive Bayes")

text = st.text_input("Enter a sentence")

if text:
    vec = vectorizer.transform([text])
    result = model.predict(vec)
    st.write("Sentiment:", result[0])
