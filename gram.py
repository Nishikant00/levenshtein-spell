import streamlit as st
import re
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import reuters, brown
from spellchecker import SpellChecker

# Function to clean and tokenize text
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Function to build n-gram language model from a corpus
def build_ngram_model(corpus, n=3):  # Using trigrams for better accuracy
    ngram_counts = Counter()
    for sentence in corpus:
        tokens = tokenize(" ".join(sentence))
        ngram_counts.update(ngrams(tokens, n))
    return ngram_counts

# Function to check grammar using n-grams
def check_grammar(text, ngram_model, n=3):  # Using trigrams for better accuracy
    tokens = tokenize(text)
    ngrams_in_text = list(ngrams(tokens, n))
    
    errors = []
    for ngram in ngrams_in_text:
        if ngram_model[ngram] == 0:
            errors.append(" ".join(ngram))
    return errors

# Build trigram model using Reuters and Brown corpus from NLTK
def get_corpus():
    return reuters.sents() + brown.sents()

def build_language_model():
    corpus = get_corpus()
    trigram_model = build_ngram_model(corpus, n=3)
    return trigram_model

# Spell checker using pyspellchecker
spell = SpellChecker()

# Streamlit App
def main():
    st.title("Lightweight N-Gram Grammar and Spell Checker")

    # Input text
    text_input = st.text_area("Enter text to check:", height=200)

    if st.button("Check Text"):
        if text_input:
            # Build trigram model once (for demonstration)
            trigram_model = build_language_model()

            # Spell checking
            tokens = tokenize(text_input)
            misspelled_words = spell.unknown(tokens)
            corrected_text = text_input
            for word in misspelled_words:
                corrected_text = corrected_text.replace(word, f"<span style='color:red'>{spell.correction(word)}</span>")

            # Grammar checking using trigrams
            grammar_errors = check_grammar(text_input, trigram_model, n=3)

            # Display results
            st.subheader("Spell Checked Text")
            st.markdown(corrected_text, unsafe_allow_html=True)

            st.subheader("Grammar Issues (Based on Trigrams)")
            if grammar_errors:
                for error in grammar_errors:
                    st.markdown(f"<p style='color:orange'>Unlikely word sequence: <strong>{error}</strong></p>", unsafe_allow_html=True)
            else:
                st.write("No trigram-based grammar issues detected.")

if __name__ == "__main__":
    main()
