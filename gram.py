import streamlit as st
import language_tool_python
from spellchecker import SpellChecker
import re

# Initialize the spell checker
spell = SpellChecker()

@st.cache_resource
def get_language_tool(language):
    return language_tool_python.LanguageTool(language)

def check_text(text, grammar_tool):
    # Check grammar
    grammar_errors = grammar_tool.check(text)
    
    # Check spelling
    words = re.findall(r'\b\w+\b', text.lower())
    misspelled = spell.unknown(words)
    
    return grammar_errors, misspelled

def highlight_errors(text, grammar_errors, misspelled):
    highlighted_text = text
    offset = 0
    
    # Highlight grammar errors
    for error in sorted(grammar_errors, key=lambda e: e.offset):
        start = error.offset + offset
        end = start + error.errorLength
        highlighted_text = (
            highlighted_text[:start] +
            f"<span style='background-color: #ADD8E6'>{highlighted_text[start:end]}</span>" +
            highlighted_text[end:]
        )
        offset += len("<span style='background-color: #ADD8E6'></span>")
    
    # Highlight spelling errors
    for word in misspelled:
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        highlighted_text = pattern.sub(
            lambda m: f"<span style='background-color: #FFB6C1'>{m.group()}</span>",
            highlighted_text
        )
    
    return highlighted_text

def get_corrected_text(text, grammar_errors):
    corrected = text
    offset = 0
    for error in sorted(grammar_errors, key=lambda e: e.offset):
        suggestion = error.replacements[0] if error.replacements else error.context
        start = error.offset + offset
        end = start + error.errorLength
        corrected = corrected[:start] + suggestion + corrected[end:]
        offset += len(suggestion) - error.errorLength
    return corrected

def main():
    st.set_page_config(page_title="Enhanced NLP Grammar and Spell Checker", layout="wide")
    
    st.title("Enhanced NLP Grammar and Spell Checker")
    
    # Sidebar for language selection
    languages = ["en-US", "en-GB", "fr", "de", "es"]
    selected_language = st.sidebar.selectbox("Select Language", languages)
    
    # Initialize the grammar checker with the selected language
    grammar_tool = get_language_tool(selected_language)
    
    # Text input
    user_input = st.text_area("Enter your text here:", height=200)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Check Grammar and Spelling"):
            if user_input:
                grammar_errors, misspelled_words = check_text(user_input, grammar_tool)
                
                # Highlight errors in the text
                highlighted_text = highlight_errors(user_input, grammar_errors, misspelled_words)
                st.subheader("Text with Highlighted Errors:")
                st.markdown(highlighted_text, unsafe_allow_html=True)
                
                # Display word count
                word_count = len(re.findall(r'\b\w+\b', user_input))
                st.info(f"Word count: {word_count}")
                
                # Display grammar suggestions
                st.subheader("Grammar Suggestions:")
                if grammar_errors:
                    for error in grammar_errors:
                        st.write(f"- {error.message}")
                        st.write(f"  Suggestion: {error.replacements[0] if error.replacements else 'No suggestion'}")
                else:
                    st.write("No grammar errors found.")
                
                # Display spelling suggestions
                st.subheader("Spelling Suggestions:")
                if misspelled_words:
                    for word in misspelled_words:
                        st.write(f"- {word}: {', '.join(spell.candidates(word))}")
                else:
                    st.write("No spelling errors found.")
                
                # Store the corrected text in session state
                st.session_state.corrected_text = get_corrected_text(user_input, grammar_errors)
            else:
                st.warning("Please enter some text to check.")
    
    with col2:
        if 'corrected_text' in st.session_state:
            st.subheader("Corrected Text:")
            st.text_area("", value=st.session_state.corrected_text, height=200)
            
            # Add a button to copy the corrected text
            if st.button("Copy Corrected Text"):
                st.write("Corrected text copied to clipboard!")
                st.code(st.session_state.corrected_text)

if __name__ == "__main__":
    main()