import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import language_tool_python

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Set page configuration
st.set_page_config(page_title="Grammar and Spell Checker", page_icon="📝", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .st-emotion-cache-1v0mbdj {
        width: 100%;
    }
    .corrected-text {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin-top: 20px;
        background-color: #f8f8f8;
    }
    .correction {
        background-color: #e8f5e9;
        text-decoration: underline;
        cursor: pointer;
    }
    .transformer-error {
        background-color: #fff9c4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pipeline('text-classification', model='textattack/roberta-base-CoLA')

@st.cache_resource
def load_language_tool():
    return language_tool_python.LanguageTool('en-US')

def check_grammar_transformers(text, classifier):
    sentences = sent_tokenize(text)
    results = []
    for sentence in sentences:
        prediction = classifier(sentence)[0]
        if prediction['label'] == 'LABEL_0':
            results.append((sentence, 'Potential error', prediction['score']))
        else:
            results.append((sentence, 'OK', prediction['score']))
    return results

def check_and_correct_text(text, tool):
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text, matches

st.title('📝 Grammar and Spell Checker')
st.markdown("Improve your writing with our advanced grammar and spell checking tool!")

# Load models
classifier = load_model()
language_tool = load_language_tool()

# Text input
text = st.text_area("Enter your text here:", height=200, max_chars=5000)

if st.button('Check and Correct Grammar and Spelling', key='check_button'):
    if text:
        # Transformer-based check
        transformer_results = check_grammar_transformers(text, classifier)
        
        # LanguageTool check and correction
        corrected_text, matches = check_and_correct_text(text, language_tool)
        
        st.subheader('Corrected Text:')
        
        # Display corrected text with highlights
        highlighted_text = corrected_text
        for match in reversed(matches):
            start, end = match.offset, match.offset + match.errorLength
            replacement = match.replacements[0] if match.replacements else match.context
            highlighted_text = (
                highlighted_text[:start] +
                f'<span class="correction" title="{match.message}">{replacement}</span>' +
                highlighted_text[end:]
            )
        
        # Add transformer-based highlights
        sentences = sent_tokenize(highlighted_text)
        transformer_highlighted_text = ""
        for sentence, (_, status, _) in zip(sentences, transformer_results):
            if status == 'Potential error':
                transformer_highlighted_text += f'<span class="transformer-error">{sentence}</span> '
            else:
                transformer_highlighted_text += sentence + ' '
        
        st.markdown(f'<div class="corrected-text">{transformer_highlighted_text}</div>', unsafe_allow_html=True)
        
        # Display correction counts
        st.info(f"Number of LanguageTool corrections: {len(matches)}")
        st.info(f"Number of potential errors detected by transformer: {sum(1 for _, status, _ in transformer_results if status == 'Potential error')}")
        
    else:
        st.warning("Please enter some text to check.")

# Add some space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>Powered by Hugging Face Transformers and LanguageTool</p>
    <p>© 2023 Grammar and Spell Checker</p>
</div>
""", unsafe_allow_html=True)