
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle5 as pickle
import nltk
from src.models.BertClassifier import BertClassifier
from nltk.sentiment import SentimentIntensityAnalyzer


# get class prediction
def pred_class(score):
    # following the advise from NLTK for Sentiment Analyisis, values betweet -0.05 and 0.05 are classified as neutral
    if score > 0.05:
        return 1
    elif score < -0.05:
        return -1
    return 0


if __name__ == '__main__':
    _root_path = '/content/drive/MyDrive/PoliticalSentimentAnalysis/'

    # load data
    with open(_root_path + 'data/tweets_emotions.pkl', 'rb') as f:
        tweets_df = pickle.load(f)

    # split up tweets for showing process and saving interim results
    tweets_df_split = np.array_split(tweets_df, 1000)
    tweets_df_labeled = pd.DataFrame()

    # Load trained BERT Model
    bert_path = _root_path + '/models/bert/model_28620'
    assert os.path.isdir(bert_path), f'Please train Bert Classifier (given dir: {bert_path})'
    print('Load Bert Classifier from:', bert_path)
    bert = BertClassifier(bert_path)

    # Init VADER
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

    # labeling
    with tqdm(total=len(tweets_df_split)) as pbar:
        for df in tweets_df_split:
            pbar.update(1)

            # BERT sentiment analysis
            df['bert'] = bert.predict(df.text.values)
            df['Polarity'] = df['bert'].apply(lambda x: x['pos'] - x['neg'])
            df['Label'] = df['polarity'].apply(pred_class)

            # VADER sentiment analysis
            df['vader'] = df.text.apply(sia.polarity_scores)
            df['vader_polarity'] = df['vader'].apply(lambda x: x['compound'])
            df['vader_label'] = df['vader_polarity'].apply(pred_class)

            tweets_df_labeled = tweets_df_labeled.append(df)

            tweets_df_labeled.to_pickle(_root_path + 'data/tweets_labeled.pkl')
            tweets_df_labeled.to_csv(_root_path + 'data/tweets_labeled.csv')
