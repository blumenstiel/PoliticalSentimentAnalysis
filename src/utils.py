from nltk.probability import FreqDist

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