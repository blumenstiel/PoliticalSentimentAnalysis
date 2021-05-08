from nltk.probability import FreqDist
import random
import pandas as pd
import numpy as np

def get_common_words(texts, count=10):
    # save all text in one list
    all_texts = list()
    for t in texts:
        all_texts.extend(t)

    # get frequency of all words
    fdist = FreqDist(w.lower() for w in all_texts if w.isalpha())

    return fdist.most_common(count)


def get_common_hashtags(texts, count):
    # save all text in one list
    all_texts = list()
    for t in texts:
        all_texts.extend(t)

    # get frequency of all words
    fdist = FreqDist(w.lower() for w in all_texts if w.startswith('#') and len(w) > 1)

    return fdist.most_common(count)


def sample_dataset(df, column, num_samples):
    """
    Get sampled dataset
    """
    # init
    samples = pd.DataFrame()
    num_labels = len(df[column].unique())

    random.seed(42)

    # get samples for each label in column of df
    for label in df[column].unique():
        label_df = df[df[column] == label]
        label_df = label_df.loc[random.sample(set(label_df.index), min(int(num_samples / num_labels), len(label_df)))]
        samples = pd.concat([samples, label_df])

    return samples