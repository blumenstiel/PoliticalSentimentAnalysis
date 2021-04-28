
from nltk import SnowballStemmer, word_tokenize, TweetTokenizer, casual_tokenize
import pandas as pd
from tqdm import tqdm

# From Lab 5 - code snippet
# Lexicon from https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm

# Stolen code from a tutorial on sentiment analysis of Harry potter
emotion_intensity_lexicon_path = 'data/NRC-Suite-of-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/' \
                                 'NRC-Emotion-Intensity-Lexicon-v1/NRC-Emotion-Intensity-Lexicon-v1.txt'

emotion_lexicon_path = 'data/NRC-Suite-of-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/' \
                       'NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'

def text_emotion(df, column, path_to_root='', lexicon='emotion_lexicon'):
    '''
    Takes a DataFrame and a specified column of text and adds 10 columns to the
    DataFrame for each of the 10 emotions in the NRC Emotion Lexicon, with each
    column containing the value of the text in that emotions
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with ten new columns
    '''

    filepath = path_to_root
    if lexicon == 'emotion_intensity_lexicon':
        filepath += emotion_intensity_lexicon_path
    elif lexicon == 'emotion_lexicon':
        filepath += emotion_lexicon_path
    else:
        filepath += lexicon

    # get emotion intensity scores from lexicon
    emolex_df = pd.read_csv(filepath,
                            names=["word", "emotion", "intensity"],
                            sep='\t', header=0)

    # create pivot table
    emolex_words = emolex_df.pivot(index='word',
                                   columns='emotion',
                                   values='intensity').reset_index()
    emolex_words.fillna(0, inplace=True)
    emolex_words['neutral'] = 0

    # get list of emotions
    emotions = emolex_words.columns.drop('word')

    # create tweet-emotion-matrix
    emo_df = pd.DataFrame(0, index=df.index, columns=emotions)

    tokenizer = TweetTokenizer()
    stemmer = SnowballStemmer("english")

    with tqdm(total=len(list(df.iterrows()))) as pbar:
        for i, row in df.iterrows():
            pbar.update(1)
            # text = word_tokenize(df.loc[i][column])
            # text = tokenizer.tokenize(df.loc[i][column])
            text = casual_tokenize(df.loc[i][column])

            for word in text:
                word = stemmer.stem(word.lower())
                if len(word) < 2:
                    continue

                emo_score = emolex_words[emolex_words.word == word]
                if not emo_score.empty:
                    emo_df.loc[i] = emo_df.loc[i] + emo_score[emotions].values[0]
                #else:
                #    emo_df.at[i, 'neutral'] += 1

    return emo_df
