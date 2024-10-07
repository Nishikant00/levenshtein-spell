import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from collections import Counter

@st.cache_resource
def load_models():
    models = {
        "vennify/t5-base-grammar-correction": {
            "tokenizer": AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction"),
            "model": AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")
        },
        "prithivida/grammar_error_correcter_v1": {
            "tokenizer": AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1"),
            "model": AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")
        },
        "Unbabel/gec-t5_small": {
            "tokenizer": AutoTokenizer.from_pretrained("Unbabel/gec-t5_small"),
            "model": AutoModelForSeq2SeqLM.from_pretrained("Unbabel/gec-t5_small")
        },
        "grammarly/coedit-large": pipeline("text2text-generation", model="grammarly/coedit-large", device=0 if torch.cuda.is_available() else -1),
        "pszemraj/flan-t5-large-grammar-synthesis": pipeline("text2text-generation", model="pszemraj/flan-t5-large-grammar-synthesis", device=0 if torch.cuda.is_available() else -1)
    }
    return models

def correct_grammar(text, models):
    corrections = {}
    for name, model in models.items():
        if isinstance(model, pipeline):
            corrected = model(text, max_length=512)[0]['generated_text']
        else:
            inputs = model["tokenizer"](text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model["model"].generate(**inputs, max_length=512)
            corrected = model["tokenizer"].decode(outputs[0], skip_special_tokens=True)
        corrections[name] = corrected
    return corrections

def ensemble_correction(corrections):
    words = [corr.split() for corr in corrections.values()]
    max_length = max(len(w) for w in words)
    padded_words = [w + [''] * (max_length - len(w)) for w in words]
    
    ensemble = []
    for word_set in zip(*padded_words):
        word_counts = Counter(word_set)
        most_common = word_counts.most_common(1)[0][0]
        if most_common:
            ensemble.append(most_common)
    
    return ' '.join(ensemble)

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
    st.set_page_config(page_title="Ensemble Grammar Checker", page_icon="✏️", layout="wide")
    st.title("✏️ Ensemble Grammar Checker")

    models = load_models()

    user_input = st.text_area("Enter your text here:", height=200)

    if st.button("Check Grammar and Spelling"):
        if user_input:
            with st.spinner("Processing with multiple models..."):
                corrections = correct_grammar(user_input, models)
                ensemble_result = ensemble_correction(corrections)
                highlighted_text = highlight_differences(user_input, ensemble_result)

            st.subheader("Ensemble Correction:")
            st.markdown(highlighted_text, unsafe_allow_html=True)

            st.subheader("Original vs Corrected:")
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("Original:", value=user_input, height=200, disabled=True)
            with col2:
                st.text_area("Corrected:", value=ensemble_result, height=200, disabled=True)

            st.subheader("Individual Model Results:")
            for model_name, correction in corrections.items():
                with st.expander(f"{model_name} Correction"):
                    st.markdown(highlight_differences(user_input, correction), unsafe_allow_html=True)
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
        This app uses an ensemble of 5 pre-trained models for grammar correction:
        - vennify/t5-base-grammar-correction
        - prithivida/grammar_error_correcter_v1
        - Unbabel/gec-t5_small
        - grammarly/coedit-large
        - pszemraj/flan-t5-large-grammar-synthesis
        Red highlights indicate original text, green highlights show corrections.
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()