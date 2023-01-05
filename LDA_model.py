#!/usr/bin/env python3

import json
import gensim
import multiprocessing
import os

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

if __name__ == "__main__":
    with open("data/lemmas.json", 'r') as f:
        abstracts = json.load(f)
    dictionary, corpus, ldamodel = lda_model(abstracts, 3)
    if not os.path.exists("data/model"):
        os.makedirs("data/model")
    dictionary.save("data/model/dictionary.dict") #save the dictionary
    gensim.corpora.mmcorpus.MmCorpus.serialize("data/model/corpus.mm", corpus) #save the corpus
    ldamodel.save("data/model/abstracts.model") #save the model
