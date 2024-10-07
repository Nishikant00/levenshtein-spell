import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
    model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")
    return tokenizer, model

def preprocess_text(text):
    corrections = {
        r'\bu\b': 'you',
        r'\br\b': 'are',
        r'\bur\b': 'your',
        r'\bthx\b': 'thanks',
        r'\bim\b': "I'm",
    }

    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    question_patterns = {
        r'\byou mad\b': 'are you mad',
    }

    for pattern, replacement in question_patterns.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = '. '.join(sentence.capitalize() for sentence in text.split('. '))
    
    return text


def correct_text(text, tokenizer, model):
    preprocessed_text = preprocess_text(text)
    
    if not preprocessed_text.strip().endswith(('.', '!', '?')):
        preprocessed_text += '.'
    
    input_ids = tokenizer.encode(preprocessed_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        input_ids, 
        max_length=512, 
        num_return_sequences=1, 
        num_beams=5,
        no_repeat_ngram_size=2,
        temperature=0.7
    )
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def is_spelling_mistake(word, corrected_word):
    return word.lower() != corrected_word.lower() and len(word) > 1

def highlight_differences(original, corrected):
    original_words = original.split()
    corrected_words = corrected.split()
    
    highlighted = []
    
    for orig_word, corr_word in zip(original_words, corrected_words):
        if orig_word.lower() != corr_word.lower():
            if is_spelling_mistake(orig_word, corr_word):
                highlighted.append(f'<span style="text-decoration: underline wavy red;">{orig_word}</span> ({corr_word})')
            else:
                highlighted.append(f'<span style="text-decoration: underline wavy blue;">{orig_word}</span> ({corr_word})')
        else:
            highlighted.append(orig_word)
    
    if len(corrected_words) > len(original_words):
        highlighted.extend([f'<span style="text-decoration: underline wavy blue;">({word})</span>' for word in corrected_words[len(original_words):]])
    elif len(original_words) > len(corrected_words):
        highlighted.extend([f'<span style="text-decoration: underline wavy blue;">{word}</span>' for word in original_words[len(corrected_words):]])
    
    return ' '.join(highlighted)

st.title("Advanced Grammar and Spelling Checker (Gramphormer)")

tokenizer, model = load_model()

user_input = st.text_area("Enter your text here:", height=200)

if st.button("Check Grammar and Spelling"):
    if user_input:
        with st.spinner("Checking your text..."):
            corrected_text = correct_text(user_input, tokenizer, model)
            highlighted_text = highlight_differences(user_input, corrected_text)
        
        st.subheader("Text with Highlights:")
        st.markdown(highlighted_text, unsafe_allow_html=True)
        
        st.subheader("Corrected Text:")
        st.text_area("You can copy the corrected text from here:", value=corrected_text, height=200)
    else:
        st.warning("Please enter some text to check.")

st.markdown("---")
st.markdown("""
    <style>
    .legend {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .legend-item {
        margin-right: 20px;
    }
    .red-underline {
        text-decoration: underline wavy red;
    }
    .blue-underline {
        text-decoration: underline wavy blue;
    }
    </style>
    <div class="legend">
        <div class="legend-item">
            <span class="red-underline">Red underline</span>: Spelling mistake
        </div>
        <div class="legend-item">
            <span class="blue-underline">Blue underline</span>: Grammar mistake
        </div>
    </div>
    """, unsafe_allow_html=True)