import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import language_tool_python

# Download necessary NLTK data
nltk.download('punkt')

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
    .result-box {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .error {
        background-color: #ffecb3;
    }
    .ok {
        background-color: #e8f5e9;
    }
    .language-tool-error {
        border-left: 3px solid #f44336;
        padding-left: 10px;
        margin-bottom: 10px;
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

def check_grammar_language_tool(text, tool):
    matches = tool.check(text)
    return matches

st.title('📝 Grammar and Spell Checker')
st.markdown("Improve your writing with our advanced grammar and spell checking tool!")

# Load models
classifier = load_model()
language_tool = load_language_tool()

# Text input
text = st.text_area("Enter your text here:", height=200, max_chars=5000)

if st.button('Check Grammar and Spelling', key='check_button'):
    if text:
        st.subheader('Results:')
        
        # Transformer-based check
        st.write("### Transformer-based Grammar Check:")
        results = check_grammar_transformers(text, classifier)
        for sentence, status, confidence in results:
            if status == 'Potential error':
                st.markdown(f"""
                <div class="result-box error">
                    <p>{sentence}</p>
                    <p><strong>Status:</strong> {status} - <strong>Confidence:</strong> {confidence:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box ok">
                    <p>{sentence}</p>
                    <p><strong>Status:</strong> {status} - <strong>Confidence:</strong> {confidence:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # LanguageTool check
        st.write("### Detailed Grammar and Spelling Check:")
        matches = check_grammar_language_tool(text, language_tool)
        if matches:
            for match in matches:
                st.markdown(f"""
                <div class="language-tool-error">
                    <p><strong style="color: #f44336;">{match.ruleId}:</strong> {match.message}</p>
                    <p><strong>Context:</strong> ...{match.context}...</p>
                    <p><strong>Suggested replacement:</strong> {match.replacements[0] if match.replacements else 'No suggestion'}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No issues found by LanguageTool.")
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