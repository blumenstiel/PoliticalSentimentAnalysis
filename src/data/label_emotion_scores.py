
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.models.EmoLex import text_emotion


if __name__ == '__main__':
    # load data
    tweets_df = pd.read_pickle('../../data/tweets_raw.pkl')

    # split up tweets for showing process and saving interim results
    tweets_df_split = np.array_split(tweets_df, 1000)
    emotion_df = pd.DataFrame()

    # labeling
    with tqdm(total=len(tweets_df_split)) as pbar:
        for df in tweets_df_split:
            pbar.update(1)

            # sentiment analysis
            emo_df = text_emotion(df, 'text', path_to_root='../../')
            emotion_df = emotion_df.append(emo_df)


            emotion_df.to_pickle('../../data/tweets_emotions.pkl')
            emotion_df.to_csv('../../data/tweets_emotions.csv')
