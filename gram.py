import streamlit as st
import re
import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter
from nltk.corpus import wordnet
import language_tool_python

nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')
word_list = set(words.words())

tool = language_tool_python.LanguageTool('en-US')
dictionary = PyDictionary()

def levenshtein_distance(word1, word2):
    dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
    
    for i in range(len(word1) + 1):
        for j in range(len(word2) + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    return dp[len(word1)][len(word2)]

def get_closest_words(word, word_list, max_suggestions=3):
    distances = [(w, levenshtein_distance(word, w)) for w in word_list if abs(len(word) - len(w)) <= 2]
    distances.sort(key=lambda x: x[1])
    return [w for w, d in distances[:max_suggestions]]

def check_spelling(text):
    tokens = word_tokenize(text.lower())
    misspelled = [word for word in tokens if word.isalpha() and word not in word_list]
    return misspelled

def highlight_mistakes(text, mistakes):
    for word in mistakes:
        text = re.sub(rf'\b({word})\b', f"**{word}**", text, flags=re.IGNORECASE)
    return text

def basic_grammar_check(text):
    matches = tool.check(text)
    grammar_issues = [f"{rule.message} at position {rule.offset}" for rule in matches]
    return grammar_issues

def get_word_meanings(word):
    synsets = wordnet.synsets(word)
    if synsets:
        return synsets[0].definition()
    return "No meaning found"

st.title("Enhanced Grammar and Spell Checker with Word Meanings")

text_input = st.text_area("Enter your text here:", height=200)

if st.button("Check"):
    if text_input:
        # Spell check
        spelling_mistakes = check_spelling(text_input)
        
        if spelling_mistakes:
            st.markdown("### Spelling Mistakes and Suggestions:")
            highlighted_text = highlight_mistakes(text_input, spelling_mistakes)
            st.markdown(highlighted_text)
            
            for word in spelling_mistakes:
                suggestions = get_closest_words(word, word_list)
                if suggestions:
                    st.write(f"Suggestions for **{word}**: {', '.join(suggestions)}")
                else:
                    st.write(f"No suggestions found for **{word}**")
                
                # Show word meanings
                st.markdown(f"**Meaning of {word}:**")
                meaning = get_word_meanings(word)
                st.write(meaning)
        
        else:
            st.success("No spelling mistakes found!")

        # Grammar check
        st.markdown("### Grammar Issues:")
        grammar_issues = basic_grammar_check(text_input)
        
        if grammar_issues:
            for issue in grammar_issues:
                st.write(f"- {issue}")
        else:
            st.success("No major grammar issues detected!")
    
    else:
        st.warning("Please enter some text to check.")
