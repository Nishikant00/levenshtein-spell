import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer
import re
import nltk
from nltk.tokenize import sent_tokenize
import time

nltk.download('punkt', quiet=True)

@st.cache_resource
def load_models():
    models = {
        "Grammar Corrector v1": {
            "tokenizer": AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1"),
            "model": AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")
        },
        "T5 Grammar": {
            "tokenizer": T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction"),
            "model": T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")
        },
        "GrammarFixer": {
            "tokenizer": AutoTokenizer.from_pretrained("Unbabel/gec-t5_small"),
            "model": AutoModelForSeq2SeqLM.from_pretrained("Unbabel/gec-t5_small")
        }
    }
    return models

def preprocess_text(text):
    corrections = {
        r'\bu\b': 'you',
        r'\br\b': 'are',
        r'\bur\b': 'your',
        r'\bthx\b': 'thanks',
        r'\bim\b': "I'm",
    }

    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    question_patterns = {
        r'\byou mad are\b': 'are you mad',
    }

    for pattern, replacement in question_patterns.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = '. '.join(sentence.capitalize() for sentence in text.split('. '))
    
    return text

def correct_text(text, tokenizer, model):
    preprocessed_text = preprocess_text(text)
    
    if not preprocessed_text.strip().endswith(('.', '!', '?')):
        preprocessed_text += '.'
    
    input_ids = tokenizer.encode(preprocessed_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        input_ids, 
        max_length=512, 
        num_return_sequences=1, 
        num_beams=5,
        temperature=0.7
    )
    
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    corrected_text = ' '.join(sent_tokenize(corrected_text))
    
    return corrected_text

def is_spelling_mistake(word, corrected_word):
    return word.lower() != corrected_word.lower() and len(word) > 1

def highlight_differences(original, corrected):
    original_words = original.split()
    corrected_words = corrected.split()
    
    highlighted = []
    
    for orig_word, corr_word in zip(original_words, corrected_words):
        if orig_word.lower() != corr_word.lower():
            if is_spelling_mistake(orig_word, corr_word):
                highlighted.append(f'<span style="text-decoration: underline wavy red;">{orig_word}</span> ({corr_word})')
            else:
                highlighted.append(f'<span style="text-decoration: underline wavy blue;">{orig_word}</span> ({corr_word})')
        else:
            highlighted.append(orig_word)
    
    if len(corrected_words) > len(original_words):
        highlighted.extend([f'<span style="text-decoration: underline wavy blue;">({word})</span>' for word in corrected_words[len(original_words):]])
    elif len(original_words) > len(corrected_words):
        highlighted.extend([f'<span style="text-decoration: underline wavy blue;">{word}</span>' for word in original_words[len(corrected_words):]])
    
    return ' '.join(highlighted)

def count_errors(original, corrected):
    original_words = original.split()
    corrected_words = corrected.split()
    
    spelling_errors = 0
    grammar_errors = 0
    
    for orig_word, corr_word in zip(original_words, corrected_words):
        if orig_word.lower() != corr_word.lower():
            if is_spelling_mistake(orig_word, corr_word):
                spelling_errors += 1
            else:
                grammar_errors += 1
    
    return spelling_errors, grammar_errors

def ensemble_correction(text, models):
    corrections = {}
    for model_name, model_data in models.items():
        corrected = correct_text(text, model_data["tokenizer"], model_data["model"])
        corrections[model_name] = corrected
    
    # Simple voting system
    words = text.split()
    final_words = []
    for i in range(len(words)):
        votes = {}
        for model_name, corrected_text in corrections.items():
            corrected_words = corrected_text.split()
            if i < len(corrected_words):
                vote = corrected_words[i]
                votes[vote] = votes.get(vote, 0) + 1
        
        final_word = max(votes, key=votes.get)
        final_words.append(final_word)
    
    return " ".join(final_words), corrections

def main():
    st.set_page_config(page_title="Ensemble Grammar and Spelling Checker", page_icon="📚", layout="wide")
    
    st.title("📚 Ensemble Grammar and Spelling Checker")

    models = load_models()

    col1, col2 = st.columns(2)

    with col1:
        user_input = st.text_area("Enter your text here:", height=300)
        
        check_button = st.button("Check Grammar and Spelling")
        
        if 'last_checked' not in st.session_state:
            st.session_state.last_checked = ''
        
        if check_button or (user_input and user_input != st.session_state.last_checked):
            if user_input:
                with st.spinner("Checking your text..."):
                    start_time = time.time()
                    ensemble_result, individual_results = ensemble_correction(user_input, models)
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    highlighted_text = highlight_differences(user_input, ensemble_result)
                    spelling_errors, grammar_errors = count_errors(user_input, ensemble_result)
                    
                    st.session_state.last_checked = user_input
                    st.session_state.ensemble_result = ensemble_result
                    st.session_state.individual_results = individual_results
                    st.session_state.highlighted_text = highlighted_text
                    st.session_state.spelling_errors = spelling_errors
                    st.session_state.grammar_errors = grammar_errors
                    st.session_state.processing_time = processing_time
            else:
                st.warning("Please enter some text to check.")

    with col2:
        if 'ensemble_result' in st.session_state:
            st.subheader("Ensemble Result:")
            st.markdown(st.session_state.highlighted_text, unsafe_allow_html=True)
            
            st.subheader("Corrected Text:")
            st.text_area("You can copy the corrected text from here:", value=st.session_state.ensemble_result, height=200)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Spelling Errors", st.session_state.spelling_errors)
            with col_b:
                st.metric("Grammar Errors", st.session_state.grammar_errors)
            with col_c:
                st.metric("Processing Time", f"{st.session_state.processing_time:.2f} seconds")
            
            st.subheader("Individual Model Results:")
            for model_name, result in st.session_state.individual_results.items():
                with st.expander(f"{model_name} Result"):
                    st.markdown(highlight_differences(user_input, result), unsafe_allow_html=True)

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
        .blue-underline {
            text-decoration: underline wavy blue;
        }
        </style>
        <div class="legend">
            <div class="legend-item">
                <span class="red-underline">Red underline</span>: Spelling mistake
            </div>
            <div class="legend-item">
                <span class="blue-underline">Blue underline</span>: Grammar mistake
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    

if __name__ == "__main__":
    main()