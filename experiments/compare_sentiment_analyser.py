
import pandas as pd
import random
import os
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from src.data.EmoLex import text_emotion
from google.cloud import language_v1

# read tweets
tweets_df = pd.read_pickle('../data/tweets_raw.pkl')

# filter english tweets
tweets_df = tweets_df[tweets_df.language == 'en']

# filter short tweets (not practical for comparing sentiment analyser)
tokenizer = TweetTokenizer()
tweets_df['length'] = tweets_df['text'].apply(lambda x: len(tokenizer.tokenize(x)))
tweets_df = tweets_df[tweets_df.length > 10]

# select test set
num_test_tweets = 1000

random_idx = random.choices(tweets_df.index, k=num_test_tweets)
test_df = tweets_df.loc[random_idx]

# VADER
sia = SentimentIntensityAnalyzer()
test_df['vader'] = test_df.text.apply(lambda t: sia.polarity_scores(t))

# TextBlob Pattern based and pretrained Naive Bayes
test_df['testblob_pattern'] = test_df.text.apply(lambda t: TextBlob(t).sentiment)
nb_analyszer = NaiveBayesAnalyzer()
test_df['testblob_nb'] = test_df.text.apply(lambda t: TextBlob(t, analyzer=nb_analyszer).sentiment)

# EmoLex
test_df['emolex'] = text_emotion(test_df, 'text', path_to_root='../').to_dict('records')

# Google Cloud Natural Language API
# IMPORTANT: To use the google cloud API, change the path to personal Google Could credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
    "/Users/benediktblumenstiel/Pycharm/PoliticalSentimentAnalysis/src/google_api_credentials.json"

client = language_v1.LanguageServiceClient()


def google_sentiment_analysis(text):
    """Get sentiment scores for text by calling the Google Cloud Natural Language API"""
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    annotations = client.analyze_sentiment(request={'document': document})
    result = dict(
        sentiment=annotations.document_sentiment.score,
        magnitude=annotations.document_sentiment.magnitude
    )
    return result


test_df['google'] = test_df.text.apply(lambda t: google_sentiment_analysis(t))

# add specific sentiment scores for each analyser
test_df['vader_score'] = test_df['vader'].apply(lambda x: x['compound'])
test_df['testblob_pattern_score'] = test_df['testblob_pattern'].apply(lambda x: x.polarity)
test_df['testblob_nb_score'] = test_df['testblob_nb'].apply(lambda x: x.p_pos - x.p_neg)
test_df['emolex_score'] = test_df['emolex'].apply(lambda x: x['positive'] - x['negative'])
test_df['emolex_score'] = test_df['emolex_score'] / test_df['length'] # normalize emolex_score with length of tweet
test_df['google_score'] = test_df['google'].apply(lambda x: x['sentiment'])


# save results
test_df.to_excel('../output/data/sentiment_analyser_comparison.xlsx')

# create correlation matrix
scores = ['vader_score', 'google_score', 'testblob_pattern_score', 'emolex_score', 'testblob_nb_score']
# sort scores after highest correlation
scores = test_df[scores].corr().sum().sort_values(ascending=False).index
corr_matrix = test_df[scores].corr()

# plot matrix
labels = ['NLTK (VADER)', 'Google Cloud\nNL API', 'TextBlob\nPattern based', 'EmoLex', 'TextBlob\nNaive Bayes']
sns.heatmap(corr_matrix, annot=True, cmap='gray', vmin=-1., vmax=1., xticklabels=labels, yticklabels=labels, fmt='.2f')
plt.tick_params(bottom=False, left=False)
plt.tight_layout()
plt.savefig('../output/figures/sentiment_analyser_correlation.png')

# plot agreement between 'vader_score' and 'google_score'
agreement = 1 - abs(test_df.vader_score - test_df.google_score)
sns.displot(agreement, binwidth=0.1, kde=True, color='gray')
plt.savefig('../output/figures/agreement_google_vader.png')

# TODO: Agreement
#  https://folk.ntnu.no/slyderse/medstat/Interrater_fullpage_9March2016.pdf
