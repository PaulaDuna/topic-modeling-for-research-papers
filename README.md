# Topic modelling for AI research paper abstracts

Python project to create a database with PubMed articles and use Gensim LDA for topic modelling.

## Project main goal

The aim of this project is to use [Biopython](https://biopython.org/) to create a database containing abstracts of PubMed articles and other relevant information based on keywords and use Latent Dirichlet Allocation (LDA) for topic modelling with [Gensim](https://radimrehurek.com/gensim/).

## Summary

This project is divided into four phases: creation of a database with abstracts and information from PubMed scientific papers; text preprocessing to prepare the articles for NLP models; NLP topic modelling; and visualization of the results.

Here is a brief summary of the steps to follow:

A) Database:

- **articles_database**: uses Biopython to search for articles in PubMed based on keywords and desired time period, fetches the articles and creates a database with their abstracts and relevant information, having provided a list of [PubMed MEDLINE display elements](https://www.nlm.nih.gov/bsd/mms/medlineelements.html) of interest.

B) Text preprocessing:

- **text_preprocessing**: performs text preprocessing for NLP techniques, which consists of converting text to lowercase letters, replacing hyphens with a space, expanding contractions, removing all but letters, tokenizing, removing stopwords and lemmatizing text. See also **most_common_word_cloud**.

C) LDA topic modelling:

- **LDA_model**: uses Gensim Latent Dirichlet Allocation (LDA) for topic modelling of preprocessed abstracts. See also **LDA_viz**.

D) Data visualization:

- **most_common_word_cloud**: creates a word cloud visualization for most common words in preprocessed texts with [wordcloud](https://amueller.github.io/word_cloud/).

- **LDA_viz**: creates a visualization for LDA topic modelling using [pyLDAvis](https://pyldavis.readthedocs.io/en/latest/index.html).

- **topic_word_cloud**: makes a list of words for word cloud visualization of most relevant terms for each topic, creates word cloud visualizations and saves files in the data folder.

## Helpful links

https://www.ncbi.nlm.nih.gov/books/NBK25501/

## Data

The dataset used in this work was created from abstracts of scientific papers published in [PubMed](https://pubmed.ncbi.nlm.nih.gov/).

* All Python code is Python3.
