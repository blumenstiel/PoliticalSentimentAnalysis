
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from google.cloud import language_v1
from src.models.EmoLex import text_emotion


# Google Cloud Natural Language API
# IMPORTANT: To use the google cloud API, change the path to personal Google Could credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
    "/Users/benediktblumenstiel/Pycharm/PoliticalSentimentAnalysis/src/google_api_credentials.json"


def google_sentiment_analysis(text, client):
    """Get sentiment scores for text by calling the Google Cloud Natural Language API"""
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    annotations = client.analyze_sentiment(request={'document': document})
    result = dict(
        sentiment=annotations.document_sentiment.score,
        magnitude=annotations.document_sentiment.magnitude
    )
    return result


def label_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sentiment Analysis of Tweets using Google Cloud Natural Language API and VADER (NLTK)
    :param df: tweets_df
    :return: labeled tweets as DataFrame
    """
    # Sentiment Analysis with Google Cloud Natural Language API
    """
    client = language_v1.LanguageServiceClient()
    df['google'] = df.text.apply(lambda t: google_sentiment_analysis(t, client))
    df['sentiment'] = df['google'].apply(lambda x: x['sentiment'])
    df['magnitude'] = df['google'].apply(lambda x: x['magnitude'])
    """

    # Sentiment Analysis with VADER
    sia = SentimentIntensityAnalyzer()
    df['vader'] = df.text.apply(lambda t: sia.polarity_scores(t))
    df['vader_score'] = df['vader'].apply(lambda x: x['compound'])

    # Add EmoLex emotions
    # df['emotions'] = text_emotion(df, 'text', path_to_root='../').to_dict('records')

    return df


if __name__ == '__main__':
    # load data
    tweets_df = pd.read_pickle('../../data/tweets_raw.pkl')

    """
    # sort values after length (potential more information, limited google resources
    tweets_df['length'] = tweets_df.text.apply(lambda x: len(x))
    tweets_df.sort_values('length', ascending=False, inplace=True)
    """

    # split up tweets for showing process and saving interim results
    tweets_df_split = np.array_split(tweets_df, 1000)
    tweets_df_labeled = pd.DataFrame()

    # labeling
    with tqdm(total=len(tweets_df_split)) as pbar:
        for df in tweets_df_split:
            pbar.update(1)

            # sentiment analysis
            df = label_tweets(df)
            tweets_df_labeled = tweets_df_labeled.append(df)
            df = None  # free memory

            tweets_df_labeled.to_pickle('../../data/tweets_labeled.pkl')
            tweets_df_labeled.to_csv('../../data/tweets_labeled.csv')
