import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Set page configuration
st.set_page_config(page_title="Grammar Checker", page_icon="📝", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .corrected-text {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin-top: 20px;
        background-color: #f8f8f8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
    model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")
    return tokenizer, model

def correct_grammar(input_text, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer(f"gec: {input_text}", return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

st.title('📝 Grammar Checker & spell checker')
st.markdown("Improve your writing with our transformer-based grammar correction tool!")

# Load model
tokenizer, model = load_model()

# Text input
text = st.text_area("Enter your text here:", height=200)

if st.button('Correct Grammar'):
    if text:
        with st.spinner('Correcting grammar...'):
            corrected_text = correct_grammar(text, tokenizer, model)
        
        st.subheader('Corrected Text:')
        st.markdown(f'<div class="corrected-text">{corrected_text}</div>', unsafe_allow_html=True)
        
    else:
        st.warning("Please enter some text to check.")

# Footer
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>© 2077 Grammar Checker</p>
</div>
""", unsafe_allow_html=True)