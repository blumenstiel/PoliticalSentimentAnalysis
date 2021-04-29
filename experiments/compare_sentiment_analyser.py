
import pandas as pd
import random
import os
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from src.models.EmoLex import text_emotion
from google.cloud import language_v1
from src.models.BertClassifier import BertClassifier, BertDataset, sample_dataset
from sklearn.model_selection import train_test_split

# OLD CODE - test on PSA tweets
"""
# read tweets
tweets_df = pd.read_pickle('../data/tweets_raw.pkl')
# filter english tweets
tweets_df = tweets_df[tweets_df.language == 'en']

# filter short tweets (not practical for comparing sentiment analyser)
tokenizer = TweetTokenizer()
tweets_df['length'] = tweets_df['text'].apply(lambda x: len(tokenizer.tokenize(x)))
tweets_df = tweets_df[tweets_df.length > 10]

# select test set
num_test_tweets = 10000

random_idx = random.choices(tweets_df.index, k=num_test_tweets)
test_df = tweets_df.loc[random_idx]
"""

# read tweets
_root_path = '/content/drive/MyDrive/PoliticalSentimentAnalysis/'

# LOAD DATA
t140 = pd.read_csv(_root_path + 'data/sentiment140/training.1600000.processed.noemoticon.csv',
                   sep=',',
                   header=None,
                   encoding='latin')

label_text = t140[[0, 5]]

# Convert labels to range -1 to 1
label_text[0] = label_text[0].apply(lambda x: 0 if x == 0 else 1)

# Assign proper column names to labels
label_text.columns = ['label', 'text']

# Assign proper column names to labels
label_text.head()

# init train and test set
num_test_tweets = 10000
test_df = sample_dataset(label_text, 'label', num_test_tweets)
train_df = label_text.drop(test_df.index, inplace=True)

# VADER
sia = SentimentIntensityAnalyzer()
test_df['vader'] = test_df.text.apply(sia.polarity_scores)

# TextBlob Pattern based and pretrained Naive Bayes
test_df['testblob_pattern'] = test_df.text.apply(lambda t: TextBlob(t).sentiment)
nb_analyszer = NaiveBayesAnalyzer()
test_df['testblob_nb'] = test_df.text.apply(lambda t: TextBlob(t, analyzer=nb_analyszer).sentiment)

# EmoLex
test_df['emolex'] = text_emotion(test_df, 'text', path_to_root=_root_path).to_dict('records')

# Google Cloud Natural Language API
# IMPORTANT: To use the google cloud API, change the path to personal Google Could credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = _root_path + 'src/google_api_credentials.json'

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


test_df['google'] = test_df.text.apply(google_sentiment_analysis)


# BERT
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_df.text.values,
                                                                                    train_df.label.values,
                                                                                    random_state=42,
                                                                                    test_size=0.1)
train_loader = BertDataset(train_inputs, train_labels).dataloader
val_loader = BertDataset(validation_inputs, validation_labels).dataloader
bert = BertClassifier()
bert.do_train(train_loader, val_loader, save_path=_root_path + 'models/bert/')

test_df['bert'] = bert.predict(test_df.text.values)

# add specific sentiment scores for each analyser
test_df['vader_score'] = test_df['vader'].apply(lambda x: x['compound'])
test_df['google_score'] = test_df['google'].apply(lambda x: x['sentiment'])
test_df['bert_score'] = test_df['bert'].apply(lambda x: x[1] - x[0])
test_df['testblob_pattern_score'] = test_df['testblob_pattern'].apply(lambda x: x.polarity)
test_df['testblob_nb_score'] = test_df['testblob_nb'].apply(lambda x: x.p_pos - x.p_neg)
test_df['emolex_score'] = test_df['emolex'].apply(lambda x: x['positive'] - x['negative'])
# normalize emolex_score with length to match format [-1, 1]
test_df['emolex_score'] = test_df['emolex_score'] / test_df.text.apply(lambda t: len(t.split(' ')))

# set negative label to -1 to match format of scores
test_df.label.apply(lambda x: -1 if x == 0 else 1)

# save results
test_df.to_excel(_root_path + 'output/data/test_sentiment_analyser.xlsx')

# create correlation matrix
scores = ['label', 'vader_score', 'google_score', 'bert_score', 'emolex_score', 'testblob_pattern_score', 'testblob_nb_score']
corr_matrix = test_df[scores].corr()

# plot matrix
labels = ['Ground Truth', 'NLTK (VADER)', 'Google Cloud\nNL API', 'BERT Classifier', 'EmoLex',
          'TextBlob\nPattern based', 'TextBlob\nNaive Bayes']
sns.heatmap(corr_matrix, annot=True, cmap='gray', vmin=-1., vmax=1., xticklabels=labels, yticklabels=labels, fmt='.2f')
plt.tick_params(bottom=False, left=False)
plt.tight_layout()
plt.savefig(_root_path + 'output/figures/sentiment_analyser_correlation.png')

# plot agreement between 'vader_score' and 'google_score'
# agreement = 1 - abs(test_df.vader_score - test_df.google_score)
# sns.displot(agreement, binwidth=0.1, kde=True, color='gray')
# plt.savefig('../output/figures/agreement_google_vader.png')

# TODO: Agreement
#  https://folk.ntnu.no/slyderse/medstat/Interrater_fullpage_9March2016.pdf
