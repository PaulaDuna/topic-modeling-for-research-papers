#!/usr/bin/env python3

import gensim
from wordcloud import WordCloud

def topics_for_word_cloud_viz(ldamodel: gensim.models.ldamulticore.LdaMulticore, num_words: int):
    """
    List of words for word cloud visualization of most relevant terms for each topic.

    Args:
        ldamodel: LDA model trained on texts.
        num_words: The number of most relevant terms for each topic to display.
    Returns:
        topics_for_viz: List of stings, one for each topic. Each string contains the most relevant terms for the corresponding topic, in decreasing order of their topic-specific probability.
    """
    topics = ldamodel.show_topics(num_words = num_words, formatted = False) #for each topic, most relevant terms and its relative weight. Ranking of terms in decreasing order of their topic-specific probability.

    topics_for_viz = []
    for i in range(len(topics)):
        text = ' '.join([word_weight[0] for word_weight in topics[i][1]])
        topics_for_viz.append(text)
    return topics_for_viz

def word_cloud_viz(text: str):
    """
    Word cloud visualization of most common words in preprocessed texts.

    Args:
        text: String that contains the most relevant terms for the corresponding topic, in decreasing order of their topic-specific probability.
    Returns:
        wordcloud: Word cloud object generated from text.
    """
    wordcloud = WordCloud(background_color = 'white', colormap = "Spectral", max_words = 50, max_font_size = 40, random_state = 42).generate(text)
    return wordcloud

if __name__ == "__main__":
    ldamodel = gensim.models.ldamulticore.LdaMulticore.load("data/model/abstracts.model")
    topics = topics_for_word_cloud_viz(ldamodel, 25)
    for i in range(len(topics)):
        wordcloud = word_cloud_viz(topics[i])
        wordcloud.to_file('data/topic_' + str(i) + '.png')
