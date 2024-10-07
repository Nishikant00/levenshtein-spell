import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def correct_grammar(text, tokenizer, model):
    prompt = f"Correct the grammar and spelling in the following text: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512, num_return_sequences=1, temperature=0.7)
    
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
    st.set_page_config(page_title="Grammar Checker and Spell Checker", page_icon="✏️", layout="wide")

    tokenizer, model = load_model()

    user_input = st.text_area("Enter your text here:", height=200)

    if st.button("Check Grammar and Spelling"):
        if user_input:
            with st.spinner("Processing with FLAN-T5..."):
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


if __name__ == "__main__":
    main()