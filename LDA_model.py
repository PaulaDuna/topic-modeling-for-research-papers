#!/usr/bin/env python3

import json
import gensim
import multiprocessing
import pyLDAvis.gensim_models

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
    dictionary, corpus, ldamodel = lda_model(abstracts, 3)
    dictionary.save("data/model/dictionary.dict") #save the dictionary
    gensim.corpora.mmcorpus.MmCorpus.serialize("data/model/corpus.mm", corpus) #save the corpus
    ldamodel.save("data/model/abstracts.model") #save the model
    viz = lda_viz(dictionary, corpus, ldamodel, 15)
    pyLDAvis.save_html(viz, 'data/ldamodel_viz.html')
