#!/usr/bin/env python3

import json
from collections import Counter
from wordcloud import WordCloud
import gensim
import multiprocessing
import pyLDAvis.gensim_models

def word_cloud_viz(texts: list, n_most_common: int):
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

def lda_model(texts: list, num_topics: int):
    """
    Latent Dirichlet Allocation (LDA) for topic modelling.

    Args:
        texts: List of lists with preprocessed words for NLP techniques.
        num_topics: The number of requested latent topics to be extracted from the training corpus.
    Returns:
        dictionary: Dictionary that encapsulates the mapping between words and their integer ids. 
        corpus: List of lists of (int, int) BoW representation of document. Each list represents a document.
        ldamodel: LDA model trained on texts.
    """
    dictionary = gensim.corpora.Dictionary(texts) #create a dictionary with words and their integer ids
    dictionary.filter_extremes(no_below = 100, no_above = 0.5) #keep tokens which are contained in at least no_below documents and in no more than no_above documents (fraction of total corpus size)
    corpus = [dictionary.doc2bow(text) for text in texts] #convert documents into the bag-of-words (BoW) format = list of (token_id, token_count) tuples
    
    cores = multiprocessing.cpu_count()
    
    ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics, id2word = dictionary, passes = 20, random_state = 123, workers = cores - 1) #train the model on the corpus
    return dictionary, corpus, ldamodel

def lda_viz(dictionary, corpus, ldamodel, terms_to_display):
    """
    Topic modelling visualization.

    Args:
        dictionary: Dictionary that encapsulates the mapping between words and their integer ids. 
        corpus: List of lists of (int, int) BoW representation of document. Each list represents a document.
        ldamodel: LDA model trained on texts.
        terms_to_display: The number of terms to display in the barcharts of the visualization.
    Returns:
        viz: A named tuple containing all the data structures required to create the visualization of the topic model.
    """
    viz = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary, R = terms_to_display)
    return viz

if __name__ == "__main__":
    with open("data/lemmas.json", 'r') as f:
        abstracts = json.load(f)
    wordcloud = word_cloud_viz(abstracts, 50)
    wordcloud.to_file('data/common_words.png')
    dictionary, corpus, ldamodel = lda_model(abstracts, 3)
    dictionary.save("data/model/dictionary.dict") #save the dictionary
    gensim.corpora.mmcorpus.MmCorpus.serialize("data/model/corpus.mm", corpus) #save the corpus
    ldamodel.save("data/model/abstracts.model") #save the model
    viz = lda_viz(dictionary, corpus, ldamodel, 15)
    pyLDAvis.save_html(viz, 'data/ldamodel_viz.html')
