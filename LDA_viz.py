#!/usr/bin/env python3

import gensim
import pyLDAvis.gensim_models

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
    dictionary = gensim.corpora.Dictionary.load("data/model/dictionary.dict")
    corpus = gensim.corpora.MmCorpus("data/model/corpus.mm")
    ldamodel = gensim.models.ldamulticore.LdaMulticore.load("data/model/abstracts.model")
    viz = lda_viz(dictionary, corpus, ldamodel, 15)
    pyLDAvis.save_html(viz, 'data/ldamodel_viz.html')
