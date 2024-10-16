import streamlit as st
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import language_tool_python

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
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

st.title('Grammar and Spell Checker')

# Load models
classifier = load_model()
language_tool = load_language_tool()

# Text input
text = st.text_area("Enter your text here:", height=200)

if st.button('Check Grammar and Spelling'):
    if text:
        st.subheader('Results:')
        
        # Transformer-based check
        st.write("Transformer-based Grammar Check:")
        results = check_grammar_transformers(text, classifier)
        for sentence, status, confidence in results:
            if status == 'Potential error':
                st.markdown(f"<span style='background-color: yellow'>{sentence}</span> - Confidence: {confidence:.2f}", unsafe_allow_html=True)
            else:
                st.write(f"{sentence} - OK (Confidence: {confidence:.2f})")
        
        # LanguageTool check
        st.write("\nDetailed Grammar and Spelling Check:")
        matches = check_grammar_language_tool(text, language_tool)
        if matches:
            for match in matches:
                st.markdown(f"<span style='color: red'>{match.ruleId}</span>: {match.message}", unsafe_allow_html=True)
                st.markdown(f"Context: ...{match.context}...")
                if match.replacements:
                    st.markdown(f"Suggested replacement: {match.replacements[0]}")
                st.write("---")
        else:
            st.write("No issues found by LanguageTool.")
    else:
        st.write("Please enter some text to check.")