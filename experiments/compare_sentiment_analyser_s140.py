
import pandas as pd
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
from google.api_core.exceptions import InvalidArgument
from src.models.BertClassifier import BertClassifier, BertDataset
from src.utils import sample_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

_root_path = '/content/drive/MyDrive/PoliticalSentimentAnalysis/'

# LOAD DATA
s140 = pd.read_csv(_root_path + 'data/sentiment140/training.1600000.processed.noemoticon.csv',
                   sep=',',
                   header=None,
                   encoding='latin')

label_text = s140[[0, 5]]

# Convert labels to range -1 to 1
label_text[0] = label_text[0].apply(lambda x: 0 if x == 0 else 1)

# Assign proper column names to labels
label_text.columns = ['label', 'text']
label_text.reset_index(drop=True, inplace=True)

# init train and test set
num_test_tweets = 10000
test_df = sample_dataset(label_text, 'label', num_test_tweets)

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

if not os.path.isdir(bert_path):
    train_df = label_text.drop(test_df.index)
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_df.text.values,
                                                                                        train_df.label.values,
                                                                                        random_state=42,
                                                                                        test_size=0.1)
    train_loader = BertDataset(train_inputs, train_labels).dataloader
    val_loader = BertDataset(validation_inputs, validation_labels).dataloader
    bert = BertClassifier()
    bert.do_train(1, train_loader, val_loader, save_path=_root_path + 'models/bert/')
else:
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
# normalize emolex_score with length of tweet
test_df['emolex_score'] = test_df['emolex_score'] / test_df.text.apply(lambda t: len(t.split(' ')))
# resize range to [-1, 1]
test_df['emolex_score'] = test_df['emolex_score'] / (max(- test_df['emolex_score'].min(), test_df['emolex_score'].max()))

# save results
test_df.to_pickle(_root_path + 'output/data/test_sentiment_analyser.pkl')
test_df.to_excel(_root_path + 'output/data/test_sentiment_analyser.xlsx')

# create correlation matrix
scores = ['label', 'bert_score', 'google_score', 'vader_score', 'emolex_score', 'testblob_pattern_score', 'testblob_nb_score']
corr_matrix = test_df[scores].corr()

# plot matrix
labels = ['Ground Truth', 'BERT Classifier', 'Google Cloud\nNL API', 'NLTK (VADER)', 'EmoLex',
          'TextBlob\nPattern based', 'TextBlob\nNaive Bayes']
sns.heatmap(corr_matrix, annot=True, cmap='gray', vmin=-1., vmax=1., xticklabels=labels, yticklabels=labels, fmt='.2f')
plt.tick_params(bottom=False, left=False)
plt.tight_layout()
plt.savefig(_root_path + 'output/figures/sentiment_analyser_correlation_s140.png')


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
test_df['testblob_pattern_pred'] = test_df['testblob_pattern_score'].apply(pred_class)
test_df['testblob_nb_pred'] = test_df['testblob_nb_score'].apply(pred_class)

# change label for negative tweets from 0 to -1
test_df.label = test_df.label.apply(lambda x: -1 if x == 0 else 1)

classifier = ['bert', 'google', 'vader', 'emolex', 'testblob_pattern', 'testblob_nb']
metrics = pd.DataFrame(index=classifier, columns=['MSE', 'Accurancy', 'F1-Score'])

for c in classifier:
    metrics.at[c, 'MSE'] = mean_squared_error(test_df.label, test_df[c + '_score'])
    metrics.at[c, 'Accurancy'] = accuracy_score(test_df.label, test_df[c + '_pred'], normalize=True)
    metrics.at[c, 'F1-Score'] = f1_score(test_df.label, test_df[c + '_pred'], labels=[-1, 1], average='micro')

metrics = metrics.astype(float).round(4)

metrics.to_excel(_root_path + 'output/data/test_sentiment_analyser_metrics.xlsx')
