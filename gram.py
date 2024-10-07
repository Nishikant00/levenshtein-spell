import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
    model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")
    return tokenizer, model

def correct_grammar(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=512)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def highlight_differences(original, corrected):
    original_words = original.split()
    corrected_words = corrected.split()
    highlighted = []

    for orig, corr in zip(original_words, corrected_words):
        if orig.lower() != corr.lower():
            highlighted.append(f'<span style="background-color: #ffcccb;">{orig}</span> → <span style="background-color: #90EE90;">{corr}</span>')
        else:
            highlighted.append(orig)

    return ' '.join(highlighted)

def main():
    st.set_page_config(page_title="Streamlined Grammar Checker", page_icon="✏️", layout="wide")
    st.title("✏️ Streamlined Grammar Checker")

    tokenizer, model = load_model()

    user_input = st.text_area("Enter your text here:", height=200)

    if st.button("Check Grammar and Spelling"):
        if user_input:
            with st.spinner("Processing..."):
                corrected_text = correct_grammar(user_input, tokenizer, model)
                highlighted_text = highlight_differences(user_input, corrected_text)

            st.subheader("Corrected Text:")
            st.markdown(highlighted_text, unsafe_allow_html=True)

            st.subheader("Original vs Corrected:")
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("Original:", value=user_input, height=200, disabled=True)
            with col2:
                st.text_area("Corrected:", value=corrected_text, height=200, disabled=True)
        else:
            st.warning("Please enter some text to check.")

    st.markdown("---")
    st.markdown("""
        <style>
        .footer {
            font-size: 0.8em;
            color: #888;
        }
        </style>
        <div class="footer">
        This app uses a pre-trained T5 model for grammar correction. 
        Red highlights indicate original text, green highlights show corrections.
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()