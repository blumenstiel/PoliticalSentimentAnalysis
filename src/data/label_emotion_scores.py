
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.models.EmoLex import text_emotion
from multiprocessing import Pool

if __name__ == '__main__':
    # load data
    _root_path = '../../'
    tweets_df = pd.read_pickle(_root_path + 'data/tweets_processed.pkl')

    # split up tweets for showing process and saving interim results
    tweets_df_split = np.array_split(tweets_df, 100)
    emotion_df = pd.DataFrame()

    # labeling
    with tqdm(total=len(tweets_df_split)) as pbar:
        for df in tweets_df_split:
            pbar.update(1)

            # sentiment analysis
            emo_df = text_emotion(df, 'tokens', path_to_root=_root_path)
            emotion_df = emotion_df.append(pd.concat([df, emo_df], axis=1))

            emotion_df.to_pickle(_root_path + 'data/tweets_emotions.pkl')
            emotion_df.to_csv(_root_path + 'data/tweets_emotions.csv')

    # get relative values for emotions
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust', 'neutral']
    # 'positive' and 'negative' are not considered as they result from combined positive and negative emotions

    sum = emotion_df[emotions].sum(axis=1)
    for e in emotions:
        emotion_df[e] = emotion_df[e] / sum

    # rename emotions for cleaner plots
    tweets_df.rename(columns={
        'anger': 'Anger',
        'anticipation': 'Anticipation',
        'joy': 'Joy',
        'trust': 'Trust',
        'fear': 'Fear',
        'surprise': 'Surprise',
        'sadness': 'Sadness',
        'disgust': 'Disgust',
        'neutral': 'Neutral',
        'negative': 'Negative',
        'positive': 'Positive'
    }, inplace=True)

    emotion_df.to_pickle(_root_path + 'data/tweets_emotions.pkl')
    emotion_df.to_csv(_root_path + 'data/tweets_emotions.csv')