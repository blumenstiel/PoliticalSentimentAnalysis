
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import time
import pickle5 as pickle

from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from src.models.EmoLex import text_emotion
from google.cloud import language_v1
from google.api_core.exceptions import InvalidArgument
from src.models.BertClassifier import BertClassifier, BertDataset
from sklearn.model_selection import train_test_split

from agreement.utils.transform import pivot_table_frequency
from agreement.metrics import cohens_kappa, krippendorffs_alpha
from agreement.utils.kernels import linear_kernel

_root_path = '/content/drive/MyDrive/PoliticalSentimentAnalysis/'

# read tweets
with open(_root_path + 'data/tweets_processed.pkl', 'rb') as f:
    tweets_df = pickle.load(f)

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
test_df['testblob'] = test_df.text.apply(lambda t: TextBlob(t).sentiment)
# convert TextBlob Class to dict
test_df['testblob'] = test_df['testblob'].apply(lambda s: dict(polarity=s[0], subjectivity=s[1]))

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
    try:
        # Catch InvalidArgument, e.g. non-english texts
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        annotations = client.analyze_sentiment(request={'document': document})
        result = dict(
            sentiment=annotations.document_sentiment.score,
            magnitude=annotations.document_sentiment.magnitude
        )
    except InvalidArgument:
        result = dict(
            sentiment=0,
            magnitude=0
        )

    return result



# iterate over test df and sleep all 600 request because of Google API rate limits
test_df['google'] = None
for i, row in test_df.iterrows():
    test_df.at[i, 'google'] = google_sentiment_analysis(row.text)

    if i != 0 and i % 500 == 0:
      time.sleep(60)

print('Finished Google API')

# BERT
bert_path = _root_path + '/models/bert/model_28620'

assert os.path.isdir(bert_path), f'Please train Bert Classifier (given dir: {bert_path})'

print('Load Bert Classifier from:', bert_path)
bert = BertClassifier(bert_path)

test_df['bert'] = bert.predict(test_df.text.values)

print('Finished BERT')

# add specific sentiment scores for each analyser
test_df['vader_score'] = test_df['vader'].apply(lambda x: x['compound'])
test_df['google_score'] = test_df['google'].apply(lambda x: x['sentiment'])
test_df['bert_score'] = test_df['bert'].apply(lambda x: x['pos'] - x['neg'])
test_df['testblob_score'] = test_df['testblob'].apply(lambda x: x['polarity'])
test_df['emolex_score'] = test_df['emolex'].apply(lambda x: x['positive'] - x['negative'])
# normalize emolex_score with length of tweet
test_df['emolex_score'] = test_df['emolex_score'] / test_df.text.apply(lambda t: len(t.split(' ')))
# resize range to [-1, 1]
test_df['emolex_score'] = test_df['emolex_score'] / (max(- test_df['emolex_score'].min(), test_df['emolex_score'].max()))

# get class prediction
def pred_class(score):
    # following the advise from NLTK for Sentiment Analyisis, values betweet -0.05 and 0.05 are classified as neutral
    if score > 0.05:
        return 1
    elif score < -0.05:
        return -1
    return 0


test_df['bert_pred'] = test_df['bert_score'].apply(pred_class)
test_df['google_pred'] = test_df['google_score'].apply(pred_class)
test_df['vader_pred'] = test_df['vader_score'].apply(pred_class)
test_df['emolex_pred'] = test_df['emolex_score'].apply(pred_class)
test_df['testblob_pred'] = test_df['testblob_score'].apply(pred_class)

# save results
test_df.to_pickle(_root_path + 'output/data/sentiment_analyser_UScongress.pkl')
test_df.to_excel(_root_path + 'output/data/sentiment_analyser_UScongress.xlsx')

# create correlation matrix
scores = ['bert_score', 'google_score', 'vader_score', 'emolex_score', 'testblob_score']
corr_matrix = test_df[scores].corr()

# set default matplotlib param
plt.rcParams["figure.figsize"] = (8, 5)
plt.rc('font', size=12)

# plot matrix
labels = ['BERT Classifier', 'Google NL API', 'NLTK (VADER)', 'EmoLex', 'TextBlob']
sns.heatmap(corr_matrix, annot=True, cmap='gray', vmin=-1., vmax=1., xticklabels=labels, yticklabels=labels, fmt='.2f')
plt.tick_params(bottom=False, left=False)
plt.tight_layout()
plt.savefig(_root_path + 'output/figures/sentiment_analyser_correlation_UScongress.png')


# agreement between classifiers
def get_kappa(n, m):
  dataset = test_df[[n, m]].stack().reset_index().values
  questions_answers_table = pivot_table_frequency(dataset[:, 0], dataset[:, 2])
  users_answers_table = pivot_table_frequency(dataset[:, 1], dataset[:, 2])
  # return krippendorffs_alpha(questions_answers_table)
  return cohens_kappa(questions_answers_table, users_answers_table, weights_kernel=linear_kernel)


preds = ['bert_pred', 'google_pred', 'vader_pred', 'emolex_pred', 'testblob_pred']

kappa = pd.DataFrame(index=preds, columns=preds)
for n in preds:
  for m in preds:
    kappa.at[n, m] = get_kappa(n, m)
kappa = kappa.astype(float)

sns.heatmap(kappa, annot=True, cmap='gray', vmin=-1., vmax=1., xticklabels=labels, yticklabels=labels, fmt='.2f')
plt.tick_params(bottom=False, left=False)
plt.tight_layout()
plt.savefig(_root_path + 'output/figures/sentiment_analyser_kappa_UScongress.png')
