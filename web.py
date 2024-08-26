import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model
import re
import os
import string
import transformer_model_l as t_model

transformer = t_model.get_transformer()
transformer.load_weights("transformer_weights.h5")

# Define custom CSS to ensure full-width layout
custom_css = """
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }
        .streamlit-expanderHeader {
            display: none;
        }
        .block-container {
            padding: 0;
            max-width: 100%;
        }
        .home-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: #e1e1e1;
            padding: 40px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            width: 100vw; /* Full viewport width */
            box-sizing: border-box;
        }
        .banner {
            flex: 1;
            display: flex;
            align-items: center;
        }
        .banner-text {
            max-width: 600px;
        }
        .banner-heading {
            font-size: 36px;
            color: #333;
        }
        .banner-subheading {
            font-size: 20px;
            color: #555;
        }
        .banner-image {
            flex: 1;
            display: flex;
            justify-content: center;
        }
        .promo-image {
            max-width: 100%;
            height: 35vw;
            border-radius: 8px;
        }
        header {
            background: #f8f8f8;
            border-bottom: 1px solid #e1e1e1;
            margin-top: 50%;
        }
        .logo {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .toggle-nav {
            display: none;
        }
        .nav-links {
            list-style: none;
            padding: 0;
            margin: 0;
            display: flex;
            gap: 20px;
        }
        .nav-links li {
            display: inline;
        }
        .nav-links a {
            text-decoration: none;
            color: #333;
            font-size: 16px;
            cursor: pointer;
        }
        .nav-links a:hover {
            text-decoration: underline;
        }
    </style>
"""

# Add custom CSS to the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# Set default page if not set
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Navigation bar using radio buttons
st.session_state.page = st.radio(
    "Navigate to:",
    ("Home", "Project", "About"),
    index=["Home", "Project", "About"].index(st.session_state.page),
    key="navigation_radio",
    horizontal=True
)

# Page rendering logic
if st.session_state.page == "Home":
    st.markdown("""
    <section>
      <div class="home-banner">
        <div class="banner">
          <div class="banner-text">
            <h1 class="banner-heading">Welcome to Our Website</h1>
            <h4 class="banner-subheading">Use Transformers for Classification, Translation, and Summarization</h4>
          </div>
        </div>
        <div class="banner-image">
          <img class="promo-image" src="https://quantdare.com/wp-content/uploads/2021/11/transformer_arch.png" alt="Promo Image">
        </div>
      </div>
    </section>
    """, unsafe_allow_html=True)

elif st.session_state.page == "Project":
    st.markdown("""
    <div style='background-color: #e1e1e1; padding: 20px; border-radius: 8px;'>
        <h1 style='text-align: center; color:rgb(37 31 31); '>Transformer Models</h1>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Text Classification")
        classification_text = st.text_area("Enter text for classification:", height=100)
        if st.button("Classify"):
            # Placeholder for classification logic
            st.write("Classifying:", "Positive")
            # Add your classification logic here

    with col2:
        st.subheader("Text Translation")
        translation_text = st.text_area("Enter text for translation:", height=100)
        if st.button("Translate"):
            # Placeholder for translation logic
            translated = t_model.decode_sequence(transformer, translation_text)
            st.write("Translating:", translated)
            # Add your translation logic here

    with col3:
        st.subheader("Text Summarization")
        summarization_text = st.text_area("Enter text for summarization:", height=100)
        if st.button("Summarize"):
            # Placeholder for summarization logic
            st.write("Summarizing:", summarization_text)
            # Add your summarization logic here


elif st.session_state.page == "About":
    st.markdown(
        """
        <div style="background-color: #e1e1e1; padding: 20px; border-radius: 10px;">
            <h1 style="color: black;">About This Website</h1>
            <h2 style="color: black;">Transformers Models from Scratch</h2>
            <p style="color: black;">
                This website features pre-trained transformer models that have been built from scratch for various natural language processing tasks:
            </p>
            <ul style="color: black; font-size: 16px;">
                <li><strong>Sentiment Classification</strong>: We developed a BERT-based model to classify movie sentiments. The model achieves an accuracy of approximately <strong>80%</strong>.</li>
                <li><strong>Hinglish to English Translation</strong>: Our text translation model demonstrates a BLEU score of <strong>77.8</strong>, ensuring high-quality translation from Hinglish to English.</li>
                <li><strong>Text Summarization</strong>: The summarization model is also performing well, effectively condensing text while maintaining meaning and context.</li>
            </ul>
            <p style="color: black;">
                These models have been carefully trained and optimized to ensure high performance across different tasks. Explore the features and see the models in action throughout the website.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
