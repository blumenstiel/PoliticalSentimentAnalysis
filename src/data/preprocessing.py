
# code based on:
# https://www.kaggle.com/menion/sentiment-analysis-with-bert-87-accuracy
# https://www.kaggle.com/sreejiths0/efficient-tweet-preprocessing

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from nltk.stem import WordNetLemmatizer


def preprocess_tweets(df: pd.DataFrame, column: str,
                      extract_information: bool = False,
                      remove_stopwords: bool = True,
                      remove_short_words: bool = True,
                      ) -> pd.DataFrame:
    """
    Preprocessing tweets
    :param df:
    :return:
    """

    # regex for hastags, mentions and urls
    hashtags = re.compile(r"^#\S+|\s#\S+")
    mentions = re.compile(r"^@\S+|\s@\S+")
    urls = re.compile(r"https?://\S+")
    whitespaces = re.compile(r"\s\s+")
    non_characters = re.compile(r"[^A-Za-z\s]+")
    short_words = re.compile(r"^\S\S?\s|\s\S\S?\s|\s\S\S?$")

    if extract_information:
        # extract hashtags and mentions
        df['hashtags'] = df[column].apply(hashtags.findall)
        df['mentions'] = df[column].apply(mentions.findall)
        # remove white spaces and "#" and "@"
        df.hashtags = df.hashtags.apply(lambda t: [h.strip()[1:] for h in t])
        df.mentions = df.mentions.apply(lambda t: [m.strip()[1:] for m in t])

    def preprocess(text):
        text = hashtags.sub('', text)
        text = mentions.sub('', text)
        text = urls.sub('', text)
        # remove all punctuation, numbers, emojis etc.
        text = non_characters.sub('', text)
        if remove_short_words:
            # remove short words (1 or 2 characters)
            text = short_words.sub(' ' ,text)
        # double white spaces, spaces on beginning or ending and lower case
        text = whitespaces.sub(' ', text).strip().lower()

        return text

    df[column] = df[column].apply(preprocess)

    # tokenize
    df['tokens'] = df.text.apply(word_tokenize)

    if remove_stopwords:
        def filter_stopwords(word):
            return word not in stopwords.words('english')

        df.tokens = df.tokens.apply(lambda t: list(filter(filter_stopwords, t)))

    lemmatizer = WordNetLemmatizer()
    df.tokens = df.tokens.apply(lambda t: [lemmatizer.lemmatize(w) for w in t])

    if extract_information:
        df['length'] = df.tokens.apply(len)

    return df


if __name__ == '__main__':
    # load data
    tweets_df = pd.read_pickle('../../data/tweets_raw.pkl')

    # filter english tweets
    if 'language' in tweets_df.columns:
        df = tweets_df[tweets_df.language == 'en']

    # split up tweets for showing process and saving interim results
    tweets_df_split = np.array_split(tweets_df, 100)
    tweets_df_processed = pd.DataFrame()

    with tqdm(total=len(tweets_df_split)) as pbar:
        for df in tweets_df_split:
            pbar.update(1)

            # preprocessing
            df = preprocess_tweets(df, 'text', extract_information=True)
            tweets_df_processed = tweets_df_processed.append(df)

    tweets_df_processed.to_pickle('../../data/tweets_processed.pkl')
    tweets_df_processed.to_csv('../../data/tweets_processed.csv')
