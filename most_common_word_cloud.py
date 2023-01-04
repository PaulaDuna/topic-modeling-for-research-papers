#!/usr/bin/env python3

import json
from collections import Counter
from wordcloud import WordCloud

def most_common_word_cloud(texts: list, n_most_common: int):
    """
    Word cloud visualization of most common words in preprocessed texts.

    Args:
        texts: List of lists with preprocessed words for NLP techniques.
        n_most_common: Number of most common words to extract from texts.
    Returns:
        wordcloud: Word cloud object generated from text.
    """
    words_list = [] #create a list with all the words from preprocessed texts
    for text in texts:
        words_list += text

    common_words = Counter(words_list).most_common(n_most_common) #create a list of (word, frequency) tuples ordered from highest to lowest frequency
    text = ' '.join([word_count[0] for word_count in common_words]) #create a str with common_words words

    wordcloud = WordCloud(background_color = 'white', colormap = "Spectral", max_words = 50, max_font_size = 40, random_state = 42).generate(text)
    return wordcloud

if __name__ == "__main__":
    with open("data/lemmas.json", 'r') as f:
        abstracts = json.load(f)
    wordcloud = most_common_word_cloud(abstracts, 50)
    wordcloud.to_file('data/common_words.png')
