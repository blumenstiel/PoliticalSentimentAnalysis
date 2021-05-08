
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import time

from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from src.models.EmoLex import text_emotion
from google.cloud import language_v1
from src.models.BertClassifier import BertClassifier, BertDataset
from sklearn.model_selection import train_test_split

_root_path = '/content/drive/MyDrive/PoliticalSentimentAnalysis/'

# read tweets
tweets_df = pd.read_pickle(_root_path + 'data/tweets_processed.pkl')

# filter short tweets (not practical for comparing sentiment analyser)
tweets_df = tweets_df[tweets_df.length > 10]

# select test set
num_test_tweets = 10000

random.seed(42)
random_idx = random.sample(list(tweets_df.index), num_test_tweets)
test_df = tweets_df.loc[random_idx]

print(f'Load test data with {len(test_df)} tweets')

# VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
test_df['vader'] = test_df.text.apply(sia.polarity_scores)

print('Finished VADER')

# TextBlob Pattern based and pretrained Naive Bayes
os.system('python -m textblob.download_corpora')
test_df['testblob_pattern'] = test_df.text.apply(lambda t: TextBlob(t).sentiment)
nb_analyszer = NaiveBayesAnalyzer()
test_df['testblob_nb'] = test_df.text.apply(lambda t: TextBlob(t, analyzer=nb_analyszer).sentiment)
# convert TextBlob Class to dict
test_df['testblob_pattern'] = test_df['testblob_pattern'].apply(lambda s: dict(polarity=s[0], subjectivity=s[1]))
test_df['testblob_nb'] = test_df['testblob_nb'].apply(lambda s: dict(classification=s[0], p_pos=s[1], p_neg=s[2]))

print('Finished TextBlob')

# EmoLex
test_df['emolex'] = text_emotion(test_df, 'text', path_to_root=_root_path).to_dict('records')

print('Finished EmoLex')

# Google Cloud Natural Language API
# IMPORTANT: To use the google cloud API, change the path to personal Google Could credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = _root_path + 'src/google_api_credentials.json'

client = language_v1.LanguageServiceClient()


def google_sentiment_analysis(text):
    # Get sentiment scores for text by calling the Google Cloud Natural Language API
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    annotations = client.analyze_sentiment(request={'document': document})
    result = dict(
        sentiment=annotations.document_sentiment.score,
        magnitude=annotations.document_sentiment.magnitude
    )
    return result


# iterate over test df and sleep all 600 request because of Google API rate limits
for i, row in test_df.iterrows():
    test_df.at[i, 'google'] = google_sentiment_analysis(row.text)

    if i != 0 and i % 600 == 0:
      time.sleep(60)

print('Finished Google API')

# BERT
bert_path = _root_path + '/models/bert/model_44720'

assert os.path.isdir(bert_path), f'Please train Bert Classifier (given dir: {bert_path})'

print('Load Bert Classifier from:', bert_path)
bert = BertClassifier(bert_path)

test_df['bert'] = bert.predict(test_df.text.values)

print('Finished BERT')

# add specific sentiment scores for each analyser
test_df['vader_score'] = test_df['vader'].apply(lambda x: x['compound'])
test_df['google_score'] = test_df['google'].apply(lambda x: x['sentiment'])
test_df['bert_score'] = test_df['bert'].apply(lambda x: x['pos'] - x['neg'])
test_df['testblob_pattern_score'] = test_df['testblob_pattern'].apply(lambda x: x['polarity'])
test_df['testblob_nb_score'] = test_df['testblob_nb'].apply(lambda x: x['p_pos'] - x['p_neg'])
test_df['emolex_score'] = test_df['emolex'].apply(lambda x: x['positive'] - x['negative'])
# normalize emolex_score with length to match format [-1, 1]
test_df['emolex_score'] = test_df['emolex_score'] / test_df.text.apply(lambda t: len(t.split(' ')))

# save results
test_df.to_pickle(_root_path + 'output/data/test_sentiment_analyser.pkl')
test_df.to_excel(_root_path + 'output/data/test_sentiment_analyser.xlsx')

# create correlation matrix
scores = ['bert_score', 'google_score', 'vader_score', 'emolex_score', 'testblob_pattern_score', 'testblob_nb_score']
corr_matrix = test_df[scores].corr()

# plot matrix
labels = ['BERT Classifier', 'Google Cloud\nNL API', 'NLTK (VADER)', 'EmoLex',
          'TextBlob\nPattern based', 'TextBlob\nNaive Bayes']
sns.heatmap(corr_matrix, annot=True, cmap='gray', vmin=-1., vmax=1., xticklabels=labels, yticklabels=labels, fmt='.2f')
plt.tick_params(bottom=False, left=False)
plt.tight_layout()
plt.savefig(_root_path + 'output/figures/sentiment_analyser_correlation_UScongress.png')

# plot agreement between 'vader_score' and 'google_score'
# agreement = 1 - abs(test_df.vader_score - test_df.google_score)
# sns.displot(agreement, binwidth=0.1, kde=True, color='gray')
# plt.savefig('../output/figures/agreement_google_vader.png')

# TODO: Agreement
#  https://folk.ntnu.no/slyderse/medstat/Interrater_fullpage_9March2016.pdf