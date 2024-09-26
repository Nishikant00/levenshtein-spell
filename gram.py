import streamlit as st
from textblob import TextBlob
from spellchecker import SpellChecker
import nltk

# Download necessary NLTK data files
nltk.download('punkt')

def check_spelling(text):
    """Check text for spelling errors and suggest corrections."""
    spell = SpellChecker()
    words = text.split()
    corrected_words = []

    for word in words:
        if spell.unknown([word]):
            corrected_words.append(spell.candidates(word).pop())  # Get the first suggestion
        else:
            corrected_words.append(word)

    corrected_text = ' '.join(corrected_words)
    return corrected_text

def check_text(text):
    """Check text for grammar and spelling."""
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    return corrected_text

def analyze_sentence_meaning(text):
    """Analyze sentence meaning and suggest rephrasing."""
    blob = TextBlob(text)
    sentiment = blob.sentiment
    
    # Generate alternative phrasing using synonyms
    sentences = blob.sentences
    suggestions = []

    for sentence in sentences:
        words = sentence.words
        rephrased = []
        for word in words:
            # Get synonyms for each word
            synonyms = set()
            for syn in word.synsets:
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name())
            # Choose the first synonym (or keep the original if no synonyms)
            rephrased.append(synonyms.pop() if synonyms else word)
        suggestions.append(" ".join(rephrased))

    return sentiment, " | ".join(suggestions)

def main():
    """Main function to run the Streamlit app."""
    st.title("Grammar, Spell, and Meaning Checker")
    
    st.write("Enter the text you want to check for grammar, spelling errors, and meaning:")
    user_input = st.text_area("Input Text", height=200)
    
    if st.button("Check"):
        if user_input:
            # Check for grammar and spelling
            grammar_corrected_text = check_text(user_input)
            spelling_corrected_text = check_spelling(user_input)
            sentiment, meaning_suggestions = analyze_sentence_meaning(user_input)
            
            st.subheader("Corrected Text (Grammar Check):")
            st.write(grammar_corrected_text)
            
            st.subheader("Corrected Text (Spelling Check):")
            st.write(spelling_corrected_text)
            
            st.subheader("Sentence Meaning Analysis:")
            st.write(f"Polarity: {sentiment.polarity} (range: -1 to 1)")
            st.write(f"Subjectivity: {sentiment.subjectivity} (range: 0 to 1)")
            st.write("Interpretation: Higher polarity indicates a positive sentiment, while lower polarity indicates a negative sentiment.")
            
            st.subheader("Suggested Rephrasings Based on Meaning:")
            st.write(meaning_suggestions)
        else:
            st.warning("Please enter some text to check.")

if __name__ == "__main__":
    main()
