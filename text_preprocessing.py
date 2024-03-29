#!/usr/bin/env python3

import pandas as pd
import contractions
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json

def text_preprocessing(texts: list) -> list:
    """
    Text preprocessing for NLP.

    Args:
        abstracts: List of strings to be preprocessed.
    Returns:
        lemmas: List of lists with preprocessed words for NLP techniques.
    """
    print('Starting text preprocessing.')
    
    text_to_lowercase = [text.lower() for text in texts if type(text) == str] #convert text to lowercase letters
    hyphen_removal = [text.replace('-', ' ') for text in text_to_lowercase] #replace hyphens with a space

    #expand contractions
    expanded_texts = []
    for text in hyphen_removal:
        expanded_words = [contractions.fix(word) for word in text.split()]
        expanded_text = ' '.join(expanded_words)
        expanded_texts.append(expanded_text)

    letters = [re.sub(r'[^a-z ]', '', text) for text in expanded_texts] #remove all but letters

    tokens = [word_tokenize(text) for text in letters] #tokenize

    #remove stopwords
    english_stopwords = stopwords.words('english')
    tokens_wo_stopwords = []
    for text in tokens:
        token_wo_stopwords = [word for word in text if word not in english_stopwords]
        tokens_wo_stopwords.append(token_wo_stopwords)

    #lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for token in tokens_wo_stopwords:
        lemma = [lemmatizer.lemmatize(word) for word in token]
        lemmas.append(lemma)

    print('Text preprocessing has finished.')

    return lemmas

if __name__ == "__main__":
    df = pd.read_csv('data/data.csv')
    abstracts = df['AB'].tolist() #list of abstracts from database
    lemmas = text_preprocessing(abstracts)
    with open("data/lemmas.json", 'w') as f:
        json.dump(lemmas, f, indent = 2) 
