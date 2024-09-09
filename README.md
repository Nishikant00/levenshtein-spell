# levenshtein-spell
A Levenshtein-based grammar checker utilizes the Levenshtein distance algorithm to detect and correct spelling and grammatical errors in text. By measuring the number of single-character edits (insertions, deletions, or substitutions) required to transform a given word into a valid word.

# Grammar and Spell Checker App

This is a Streamlit-based application for checking spelling mistakes and basic grammar issues in text. It provides spell suggestions using the Levenshtein Distance algorithm and highlights potential grammar issues.

## Features

- **Spell Checking:** Identifies misspelled words and provides suggestions based on the Levenshtein Distance.
- **Grammar Checking:** Performs basic grammar checks to flag possible issues.
- **Interactive UI:** Highlights spelling mistakes and suggests corrections interactively.

## Requirements

- Python 3.x
- `streamlit`
- `nltk`
