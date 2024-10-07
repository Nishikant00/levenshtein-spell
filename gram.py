import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
    model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")
    return tokenizer, model

def preprocess_text(text):
    # Convert common abbreviations
    text = re.sub(r'\bu\b', 'you', text, flags=re.IGNORECASE)
    text = re.sub(r'\br\b', 'are', text, flags=re.IGNORECASE)
    text = re.sub(r'\bur\b', 'your', text, flags=re.IGNORECASE)
    text = re.sub(r'\bthx\b', 'thanks', text, flags=re.IGNORECASE)
    
    # Ensure proper capitalization
    text = '. '.join(sentence.capitalize() for sentence in text.split('. '))
    
    return text

def correct_text(text, tokenizer, model):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Ensure the text ends with proper punctuation
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

def highlight_differences(original, corrected):
    words1 = original.split()
    words2 = corrected.split()
    highlighted = []
    
    for word1, word2 in zip(words1, words2):
        if word1.lower() != word2.lower():
            highlighted.append(f'<span style="text-decoration: underline wavy red;">{word1}</span>')
            highlighted.append(f'<span style="text-decoration: underline wavy green;">{word2}</span>')
        else:
            highlighted.append(word2)
    
    # Add any remaining words
    if len(words2) > len(words1):
        highlighted.extend([f'<span style="text-decoration: underline wavy green;">{word}</span>' for word in words2[len(words1):]])
    elif len(words1) > len(words2):
        highlighted.extend([f'<span style="text-decoration: underline wavy red;">{word}</span>' for word in words1[len(words2):]])
    
    return ' '.join(highlighted)

st.title("Grammar and Spelling Checker")

tokenizer, model = load_model()

user_input = st.text_area("Enter your text here:", height=200)

if st.button("Check Grammar and Spelling"):
    if user_input:
        with st.spinner("Checking your text..."):
            corrected_text = correct_text(user_input, tokenizer, model)
            highlighted_text = highlight_differences(user_input, corrected_text)
        
        st.subheader("Original Text with Highlights:")
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
    .green-underline {
        text-decoration: underline wavy green;
    }
    </style>
    <div class="legend">
        <div class="legend-item">
            <span class="red-underline">Red underline</span>: Original text
        </div>
        <div class="legend-item">
            <span class="green-underline">Green underline</span>: Corrected text
        </div>
    </div>
    """, unsafe_allow_html=True)