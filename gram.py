import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import difflib

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
    model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")
    return tokenizer, model

def correct_text(text, tokenizer, model):
    # Ensure the text ends with proper punctuation
    if not text.strip().endswith(('.', '!', '?')):
        text += '.'
    
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        input_ids, 
        max_length=512, 
        num_return_sequences=1, 
        num_beams=5,
        no_repeat_ngram_size=2  # Prevent repetition of 2-grams
    )
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def highlight_differences(original, corrected):
    d = difflib.Differ()
    diff = list(d.compare(original.split(), corrected.split()))
    
    highlighted = []
    for word in diff:
        if word.startswith('  '):
            highlighted.append(word[2:])
        elif word.startswith('- '):
            highlighted.append(f'<span style="background-color: #ffcccb;">{word[2:]}</span>')
        elif word.startswith('+ '):
            highlighted.append(f'<span style="background-color: #90EE90;">{word[2:]}</span>')
    
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
    .red-highlight {
        background-color: #ffcccb;
    }
    .green-highlight {
        background-color: #90EE90;
    }
    </style>
    <div class="legend">
        <div class="legend-item">
            <span class="red-highlight">Red highlight</span>: Removed or corrected
        </div>
        <div class="legend-item">
            <span class="green-highlight">Green highlight</span>: Added or corrected
        </div>
    </div>
    """, unsafe_allow_html=True)